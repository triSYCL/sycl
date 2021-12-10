//===- PrepareSyclChessOpt.cpp                               ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Prepare sycl chess device code for Optimizations.
//
// ===---------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/SYCL/PrepareSyclChessOpt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"

#include "DownGradeUtils.h"

using namespace llvm;

namespace {

struct PrepareSyclChessOpt : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  PrepareSyclChessOpt() : ModulePass(ID) {}

  /// Make non-kernel functions and global variables private.
  /// Because only kernels are externally visible.
  void turnNonKernelsIntoPrivate(Module &M) {
    for (GlobalObject &G : M.global_objects())
      if (auto *F = dyn_cast<Function>(&G))
        if (F->getCallingConv() != CallingConv::SPIR_KERNEL &&
            !G.isDeclaration())
          G.setLinkage(llvm::GlobalValue::PrivateLinkage);
  }

  /// We give an attribute with a unique id to each kernels such
  /// that the function merger pass does not merge them.
  /// The first level of call in the kernel (which is introduced by the
  /// runtime) will be inlined into the kernel function.
  void prepareForMerging(Module &M) {
    int id = 0;
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G)) {
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL) {
          F->addFnAttr("unmergable-kernel-id",
                       StringRef(llvm::to_string(id++)));
        }
      }
    }
  }

  /// Removes SPIR_FUNC/SPIR_KERNEL calling conventions from functions and
  /// replace them with the default C calling convention for now
  /// TODO: factorize common part with the FPGA flow.
  void modifySPIRCallingConv(Module &M) {
    for (auto &F : M.functions()) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL ||
          F.getCallingConv() == CallingConv::SPIR_FUNC) {
        if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
          F.addFnAttr("chess_sycl_kernel");

        // C - The default llvm calling convention, compatible with C.  This
        // convention is the only calling convention that supports varargs calls.
        // As with typical C calling conventions, the callee/caller have to
        // tolerate certain amounts of prototype mismatch.
        // Calling Convention List For Reference:
        // https://llvm.org/doxygen/CallingConv_8h_source.html#l00029
        // Changing top level function definition/declaration, not call sites
        F.setCallingConv(CallingConv::C);

        // setCallingConv on the function won't change all the call sites,
        // we must replicate the calling convention across it's Uses. Another
        // method would be to go through each basic block and check each
        // instruction, but this seems more optimal

        /// Go through every users to find call sites and update them
        SmallVector<User *, 8> Stack;
        SmallSet<User *, 16> Set;
        Stack.push_back(&F);
        Set.insert(&F);
        while (!Stack.empty()) {
          auto *V = Stack.pop_back_val();
          if (auto *CB = dyn_cast<CallBase>(V)) {
            CB->setCallingConv(CallingConv::C);
            continue;
          }
          for (auto U : V->users())
            if (Set.insert(U).second)
              Stack.push_back(U);
        }
      }
    }
  }

  /// This will collect every constructor and destructor calls and put them
  /// into specific runtime functions that will be invoked at the right time by
  /// the runtime.
  void handleGlobals(Module &M) {
    /// Chess cannot handle llvm.global_ctors so we remove it
    if (GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors"))
      GV->eraseFromParent();
    /// Find the runtime function that will be called to run constructors
    if (Function *OurF = M.getFunction("__cxx_global_var_ctor")) {
      for (Function &F : M.getFunctionList()) {
        /// Find every constructors and add them to __cxx_global_var_ctor
        if (F.getName().startswith("__cxx_global_var_init"))
          CallInst::Create(F.getFunctionType(), &F, "",
                           OurF->getEntryBlock().getTerminator());
      }
    }
    /// Find the runtime function that will be called to run destructors
    if (Function *OurF = M.getFunction("__cxx_global_var_dtor"))
      if (Function *AtExit = M.getFunction("__cxa_atexit")) {
        /// Go through every calls to at exit
        for (User *U : AtExit->users()) {
          /// Add function they asked to call to __cxx_global_var_dtor
          Function *F = cast<Function>(getUnderlyingObject(U->getOperand(0)));
          GlobalVariable *GV = cast<GlobalVariable>(getUnderlyingObject(U->getOperand(1)));
          CallInst::Create(F->getFunctionType(), F, {GV}, "",
                           OurF->getEntryBlock().getTerminator());
        }
      }
  }

  /// Make sure that a symbol is visible in the resulting binary
  void makeVisible(Module &M, StringRef Symbol) {
    auto* GV = M.getNamedGlobal(Symbol);
    if (!GV)
      return;
    GV->setComdat(nullptr);
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
    GV->setDSOLocal(false);
    GV->setLinkage(llvm::GlobalValue::ExternalLinkage);
  }

  bool runOnModule(Module &M) override {
    /// Mark invisible globals as private
    turnNonKernelsIntoPrivate(M);
    /// Prevent the function merger from merging kernels
    prepareForMerging(M);
    /// Accumulate constructors and destructors into specific runtime functions
    handleGlobals(M);
    /// Keep kernel_lambda_capture visible because the runtime needs it
    makeVisible(M, "kernel_lambda_capture");
    /// We are not SPIR so make all calling convention C
    modifySPIRCallingConv(M);

    llvm::replaceFunction(M, "abort", "_Z13finish_kernelv");
    return true;
  }
};
}

namespace llvm {
void initializePrepareSyclChessOptPass(PassRegistry &Registry);
}

INITIALIZE_PASS(PrepareSyclChessOpt, "preparechess",
  "prepare SYCL chess device code to optimizations", false, false)
ModulePass *llvm::createPrepareSyclChessOptPass() {return new PrepareSyclChessOpt();}

char PrepareSyclChessOpt::ID = 0;
