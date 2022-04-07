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
#include <iostream>
#include <regex>
#include <string>

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Attributes.h"
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

  inline bool isKernel(Function &F) {
    // Kernel are first detected with the SPIR_KERNEL CC.
    // After a first run of this pass in case of HLS flow,
    // this CC is replaced and kernels are marked with an
    // fpga.top.func attribute.
    // (See setHLSCallingConvention)
    return (F.getCallingConv() == CallingConv::SPIR_KERNEL ||
            F.hasFnAttribute("fpga.top.func"));
  }

  void turnNonKernelsIntoPrivate(Module &M) {
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G))
        if (isKernel(*F))
          continue;
      if (G.isDeclaration())
        continue;
      G.setComdat(nullptr);
      G.setLinkage(llvm::GlobalValue::PrivateLinkage);
    }
  }

  /// Add the flatten attribute to all kernel and noinline
  /// functions, in oder for all non-kernel and non-noinline
  /// functions to be inlined
  void markKernelandNoInlineForFlattening(Module &M) {
    for (auto &F : M.functions()) {
      if (isKernel(F) || F.hasFnAttribute(Attribute::NoInline)) {
        F.addFnAttr("flatten");
      }
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
      auto FnAttr = F->getAttributes().getFnAttrs();
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
  /// a functor that wraps the kernel in a function which is annotated
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

  struct CheckUnsupportedBuiltinsVisitor
      : public llvm::InstVisitor<CheckUnsupportedBuiltinsVisitor> {
    void visitCallInst(CallInst &I) {
      auto *F = I.getCalledFunction();
      if (llvm::demangle(std::string(F->getName()))
              .rfind("__spir_ocl_get", 0) == 0) {
        std::cerr << "SYCL_VXX_UNSUPPORTED_SPIR_BUILTINS" << std::endl;
      }
    }
  };

  void signalUnsupportedSPIRBuiltins(Module &M) {
    CheckUnsupportedBuiltinsVisitor CUBV{};
    for (auto &F : M.functions()) {
      CUBV.visit(F);
    }
  }

  bool runOnModule(Module &M) override {
    // When using the HLS flow instead of SPIR default
    bool SyclHLSFlow = Triple(M.getTargetTriple()).isXilinxHLS();
    unwrapFPGAProperties(M);
    turnNonKernelsIntoPrivate(M);
    if (SyclHLSFlow) {
      setHLSCallingConvention(M);
      signalUnsupportedSPIRBuiltins(M);
      if (ClearSpir)
        cleanSpirBuiltins(M);
    } else {
      setCallingConventions(M);
    }
    lowerArrayPartition(M);
    if (!SyclHLSFlow)
      forceInlining(M);
    else
      markKernelandNoInlineForFlattening(M);
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
