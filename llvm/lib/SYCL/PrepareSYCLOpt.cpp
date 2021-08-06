//===- PrepareSYCLOpt.cpp - Perform some code janitoring -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Prepare device code for Optimizations.
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/SYCL/PrepareSYCLOpt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

cl::opt<bool> ClearSpir("sycl-prepare-clearspir", cl::Hidden, cl::init(false));

struct PrepareSYCLOpt : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  PrepareSYCLOpt() : ModulePass(ID) {}

  void turnNonKernelsIntoPrivate(Module &M) {
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G))
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL ||
            F->hasFnAttribute("fpga.top.func"))
          continue;
      if (G.isDeclaration())
        continue;
      G.setComdat(nullptr);
      G.setLinkage(llvm::GlobalValue::PrivateLinkage);
    }
  }

  void setHLSCallingConvention(Module &M) {
    for (Function &F : M.functions()) {
      // If the function is a kernel or an intrinsic, keep the current CC
      if (F.hasFnAttribute("fpga.top.func") || F.isIntrinsic()) {
        continue;
      }
      // Annotate kernels for HLS backend being able to identify them
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
        assert(F.use_empty());
        F.addFnAttr("fpga.top.func", F.getName());
        F.addFnAttr("fpga.demangled.name", F.getName());
        F.setCallingConv(CallingConv::C);
        F.setLinkage(llvm::GlobalValue::ExternalLinkage);
      } else {
        // We need to call intrinsic with SPIR_FUNC calling conv
        // for correct linkage with Vitis SPIR builtins lib
        auto cc = (ClearSpir) ? CallingConv::C : CallingConv::SPIR_FUNC;
        F.setCallingConv(cc);
        for (Value *V : F.users()) {
          if (auto *Call = dyn_cast<CallBase>(V))
            Call->setCallingConv(cc);
        }
      }
    }
  }

  void setCallingConventions(Module &M) {
    for (Function &F : M.functions()) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
        assert(F.use_empty());
        continue;
      }
      if (F.isIntrinsic())
        continue;
      F.setCallingConv(CallingConv::SPIR_FUNC);
      for (Value *V : F.users()) {
        if (auto *Call = dyn_cast<CallBase>(V))
          Call->setCallingConv(CallingConv::SPIR_FUNC);
      }
    }
  }

  /// At this point in the pipeline Annotations intrinsic have all been
  /// converted into what they need to be. But they can still be present and
  /// have pointer on pointer as arguments which v++ can't deal with.
  void removeAnnotations(Module &M) {
    SmallVector<Instruction *, 16> ToRemove;
    for (Function &F : M.functions())
      if (F.getIntrinsicID() == Intrinsic::annotation ||
          F.getIntrinsicID() == Intrinsic::ptr_annotation ||
          F.getIntrinsicID() == Intrinsic::var_annotation)
        for (User *U : F.users())
          if (auto *I = dyn_cast<Instruction>(U))
            ToRemove.push_back(I);
    for (Instruction *I : ToRemove)
      I->eraseFromParent();
    GlobalVariable *Annot = M.getGlobalVariable("llvm.global.annotations");
    if (Annot)
      Annot->eraseFromParent();
  }

  /// This will change array partition such that after the O3 pipeline it
  /// matched very closely what v++ generates.
  /// This will change the type of the alloca referenced by the array partition
  /// into an array. and change the argument received by xlx_array_partition
  /// into a pointer on an array.
  void lowerArrayPartition(Module &M) {
    Function *Func = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    for (Use &U : Func->uses()) {
      auto *Usr = dyn_cast<CallBase>(U.getUser());
      if (!Usr)
        continue;
      if (!Usr->getOperandBundle("xlx_array_partition"))
        continue;
      Use &Ptr = U.getUser()->getOperandUse(0);
      Value *Obj = getUnderlyingObject(Ptr);
      if (!isa<AllocaInst>(Obj))
        return;
      auto *Alloca = cast<AllocaInst>(Obj);
      auto *Replacement =
          new AllocaInst(Ptr->getType()->getPointerElementType(), 0,
                         ConstantInt::get(Type::getInt32Ty(M.getContext()), 1),
                         Align(128), "");
      Replacement->insertAfter(Alloca);
      Instruction *Cast = BitCastInst::Create(Instruction::BitCast, Replacement,
                                              Alloca->getType());
      Cast->insertAfter(Replacement);
      Alloca->replaceAllUsesWith(Cast);
      Value *Zero = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
      Instruction *GEP =
          GetElementPtrInst::Create(nullptr, Replacement, {Zero});
      GEP->insertAfter(Cast);
      Ptr.set(GEP);
    }
  }

  void forceInlining(Module &M) {
    for (auto &F : M.functions()) {
      if (F.isDeclaration() || F.getCallingConv() == CallingConv::SPIR_KERNEL)
        continue;
      F.addFnAttr(Attribute::AlwaysInline);
    }
  }

  void cleanSpirBuiltins(Module &M) {
    /// Find function
    auto *spirid = M.getFunction("llvm.spir.get.global.id.i64");
    if (spirid != nullptr && spirid->isDeclaration()) {
      /// Create replacement
      auto *replacement = ConstantInt::get(spirid->getReturnType(), 1);
      for (auto *user : spirid->users())
        if (auto *call = dyn_cast<CallBase>(user)) {
          /// Replace calls by constant
          call->replaceAllUsesWith(replacement);
          call->eraseFromParent();
        }
      assert(spirid->use_empty());
      /// Erase the function from the module.
      spirid->eraseFromParent();
    }
  }

  /// Visit call instruction to check if the called function is a property
  /// wrapper, i.e. a function that just call another function and has
  /// interesting HLS annotation.
  /// When a property wrapper is found, it moves its annotation to the caller 
  /// and inline it.
  struct UnwrapperVisitor : public llvm::InstVisitor<UnwrapperVisitor> {
    void visitCallInst(CallInst &I) {
      auto *ParentF = I.getFunction();
      auto *F = I.getCalledFunction();
      if (!F->hasFnAttribute("fpga.propertywrapper"))
        return;
      // We have a property wrapper.
      // First, unwrap all wrapper inside F
      visit(*F);

      // Now copy fpga attributes to parent
      auto FnAttr = F->getAttributes().getFnAttributes();
      for (auto &Attr : FnAttr) {
        if (Attr.isStringAttribute()) {
          StringRef AttrKind = Attr.getKindAsString();
          if (AttrKind.startswith("fpga.") &&
              AttrKind != "fpga.propertywrapper") {
            ParentF->addFnAttr(Attr);
          }
        }
      }
      // And inline the wrapper inside the caller
      llvm::InlineFunctionInfo IFI;
      llvm::InlineFunction(I, IFI);
    }
  };
  
  /// Kernel level property are marked using a KernelDecorator, 
  /// a functor that wrap the kernel in a function which is annotated 
  /// in a way that is later transformed to HLS compatible annotations.
  /// 
  /// This function inline the wrapping (decorator) function while
  /// preserving the HLS annotations (by annotating the caller). 
  void unwrapFPGAProperties(Module &M) {
    UnwrapperVisitor UWV{};
    for (auto &F : M.functions()) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
        UWV.visit(F);
      }
    }
  }

  bool runOnModule(Module &M) override {
    // When using the HLS flow instead of SPIR default
    bool SyclHLSFlow = Triple(M.getTargetTriple()).isXilinxHLS();
    unwrapFPGAProperties(M);
    turnNonKernelsIntoPrivate(M);
    if (SyclHLSFlow) {
      setHLSCallingConvention(M);
      if (ClearSpir)
        cleanSpirBuiltins(M);
    } else {
      setCallingConventions(M);
    }
    lowerArrayPartition(M);
    if (ClearSpir)
      removeAnnotations(M);
    if (!SyclHLSFlow)
      forceInlining(M);
    return true;
  }
};
} // namespace

namespace llvm {
void initializePrepareSYCLOptPass(PassRegistry &Registry);
}

INITIALIZE_PASS(PrepareSYCLOpt, "preparesycl",
                "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createPrepareSYCLOptPass() { return new PrepareSYCLOpt(); }

char PrepareSYCLOpt::ID = 0;
