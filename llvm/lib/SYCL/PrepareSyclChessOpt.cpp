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

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/SYCL/PrepareSyclChessOpt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;

namespace {

struct PrepareSyclChessOpt : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  PrepareSyclChessOpt() : ModulePass(ID) {}

  void turnNonKernelsIntoPrivate(Module &M) {
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G)) {
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL)
          continue;
        if (G.isDeclaration())
          continue;
        G.setLinkage(llvm::GlobalValue::PrivateLinkage);
      }
    }
  }

  void prepareForMerging(Module &M) {
    /// we give a attribute with a unique id to every kenrels such
    /// that mergefunc dont merge them.
    /// the first level of call in the kernel (which is introduced by the
    /// runtime) Will be inlined into the kernel function.
    int id = 0;
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G)) {
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL) {
          F->addFnAttr("unmergable-kernel-id",
                       StringRef(llvm::to_string(id++)));
          for (auto &I : instructions(F))
            if (auto *CB = dyn_cast<CallBase>(&I))
              CB->getCalledFunction()->addFnAttr(Attribute::AlwaysInline);
        }
      }
    }
  }

  struct MakeVolatileVisitor : InstVisitor<MakeVolatileVisitor> {
    void visitLoadInst(LoadInst &I) { I.setVolatile(true); }
    void visitStoreInst(StoreInst &I) { I.setVolatile(true); }
  };

  /// Make every store or load of unknow volatile.
  /// TODO we could only change lod and store of unknow provenance.
  void makeVolatile(Module &M) {
    MakeVolatileVisitor Visitor;
    Visitor.visit(M);
  }


  /// Removes SPIR_FUNC/SPIR_KERNEL calling conventions from functions and
  /// replace them with the default C calling convention for now
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
        // Changing top level function defintiion/declaration, not call sites
        F.setCallingConv(CallingConv::C);

        // setCallingConv on the function won't change all the call sites,
        // we must replicate the calling convention across it's Uses. Another
        // method would be to go through each basic block and check each
        // instruction, but this seems more optimal
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

  void handleGlobals(Module &M) {
    /// Chess cannot handle llvm.global_ctors so we remove it.
    if (GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors")) {
      GV->eraseFromParent();
    }
    if (Function *OurF = M.getFunction("__cxx_global_var_ctor")) {
      for (Function &F : M.getFunctionList()) {
        if (F.getName().startswith("__cxx_global_var_init"))
          CallInst::Create(F.getFunctionType(), &F, "",
                           OurF->getEntryBlock().getTerminator());
      }
    }
    if (Function *OurF = M.getFunction("__cxx_global_var_dtor"))
      if (Function *AtExit = M.getFunction("__cxa_atexit")) {
        for (User *U : AtExit->users()) {
          Function *F = cast<Function>(getUnderlyingObject(U->getOperand(0)));
          GlobalVariable *GV = cast<GlobalVariable>(getUnderlyingObject(U->getOperand(1)));
          CallInst::Create(F->getFunctionType(), F, {GV}, "",
                           OurF->getEntryBlock().getTerminator());
        }
      }
  }

  bool runOnModule(Module &M) override {
    turnNonKernelsIntoPrivate(M);
    prepareForMerging(M);
    handleGlobals(M);
    // This has to be done before changing the calling convention
    modifySPIRCallingConv(M);
    // makeVolatile(M);
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
