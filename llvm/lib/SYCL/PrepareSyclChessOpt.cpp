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

  void makeKernelsUnmergable(Module &M) {
    /// we give a attribute with a unique id to every kenrels such that
    /// mergefunc dont merge them.
    int id = 0;
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G)) {
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL)
          F->addFnAttr("unmergable-kernel-id",
                       StringRef(llvm::to_string(id++)));
      }
    }
  }

  bool runOnModule(Module &M) override {
    turnNonKernelsIntoPrivate(M);
    makeKernelsUnmergable(M);
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
