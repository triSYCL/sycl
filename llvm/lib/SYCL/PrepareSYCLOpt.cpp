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
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/SYCL/PrepareSYCLOpt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#include "SYCLUtils.h"

using namespace llvm;

namespace {

cl::opt<bool> AfterO3("sycl-prepare-after-O3", cl::Hidden, cl::init(false));

struct PrepareSYCLOptState {

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
      if (G.getName() == "llvm.global_ctors" || G.isDeclaration())
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
      if (sycl::isKernelFunc(&F)) {
        assert(F.use_empty());
        sycl::annotateKernelFunc(&F);
      } else {
        // We need to call intrinsic with SPIR_FUNC calling conv
        // for correct linkage with Vitis SPIR builtins lib
        auto cc = (AfterO3) ? CallingConv::C : CallingConv::SPIR_FUNC;
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
      if (sycl::isKernelFunc(&F)) {
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

  void forceInlining(Module &M) {
    for (auto &F : M.functions()) {
      if (F.isDeclaration() || sycl::isKernelFunc(&F))
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
      if (sycl::isKernelFunc(&F)) {
        UWV.visit(F);
      }
    }
  }

  struct CheckUnsupportedBuiltinsVisitor
      : public llvm::InstVisitor<CheckUnsupportedBuiltinsVisitor> {
    void visitCallInst(CallInst &I) {
      auto *F = I.getCalledFunction();
      if (F && llvm::demangle(std::string(F->getName()))
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

  /// Lower memory intrinsics into a simple loop of load and/or store
  /// Note: the default intrinsic lowering is what we need for HLS because the
  /// LowerToNonI8Type flag is used by sycl_vxx
  void lowerMemIntrinsic(Module &M) {
    TargetTransformInfo TTI(M.getDataLayout());
    SmallVector<Instruction *, 16> ToProcess;
    for (auto &F : M.functions())
      for (auto &I : instructions(F))
        if (auto *CI = dyn_cast<CallBase>(&I))
          ToProcess.push_back(&I);
    for (auto *I : ToProcess) {
      if (auto *CI = dyn_cast<CallBase>(I)) {
        if (MemCpyInst *Memcpy = dyn_cast<MemCpyInst>(CI))
          expandMemCpyAsLoop(Memcpy, TTI);
        else if (MemMoveInst *Memmove = dyn_cast<MemMoveInst>(CI))
          expandMemMoveAsLoop(Memmove);
        else if (MemSetInst *Memset = dyn_cast<MemSetInst>(CI))
          expandMemSetAsLoop(Memset);
        else
          continue;
        CI->eraseFromParent();
      }
    }
  }

  bool runOnModule(Module &M) {
    // When using the HLS flow instead of SPIR default
    bool SyclHLSFlow = Triple(M.getTargetTriple()).isXilinxHLS();
    unwrapFPGAProperties(M);
    turnNonKernelsIntoPrivate(M);
    lowerMemIntrinsic(M);
    if (SyclHLSFlow) {
      setHLSCallingConvention(M);
      signalUnsupportedSPIRBuiltins(M);
      if (AfterO3)
        cleanSpirBuiltins(M);
    } else {
      setCallingConventions(M);
    }
    if (!SyclHLSFlow)
      forceInlining(M);
    else
      markKernelandNoInlineForFlattening(M);
    return true;
  }
};

void runPrepareSYCLOpt(Module &M) {
  PrepareSYCLOptState State;
  State.runOnModule(M);
}

} // namespace

PreservedAnalyses PrepareSYCLOptPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  runPrepareSYCLOpt(M);
  return PreservedAnalyses::none();
}

struct PrepareSYCLOptLegacy : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  PrepareSYCLOptLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    runPrepareSYCLOpt(M);
    return true;
  }
};

namespace llvm {
void initializePrepareSYCLOptLegacyPass(PassRegistry &Registry);
}

INITIALIZE_PASS(PrepareSYCLOptLegacy, "preparesycl",
                "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createPrepareSYCLOptLegacyPass() { return new PrepareSYCLOptLegacy(); }

char PrepareSYCLOptLegacy::ID = 0;
