//===- XOCCIRDowngrader.cpp                                      ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Erases and modifies IR incompatabilities with XOCC backend
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/SYCL/XOCCIRDowngrader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// As a rule of thumb, if a pass to downgrade a part of the IR is added it
/// should have the LLVM version and date of patch/patch (if possible) that it
/// was added in so it can eventually be removed as xocc catches up
struct XOCCIRDowngrader : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  XOCCIRDowngrader() : ModulePass(ID) {}

  /// Removes immarg (immutable arg) bitcode attribute that is applied to
  /// function parameters. It was added in LLVM-9 (D57825), so as xocc catches
  /// up it can be removed
  void removeImmarg(Module &M) {
    for (auto &F : M.functions()) {
      int i = 0;
      for (auto &P : F.args()) {
          if (P.hasAttribute(llvm::Attribute::ImmArg)
              || F.hasParamAttribute(i, llvm::Attribute::ImmArg)) {
              P.removeAttr(llvm::Attribute::ImmArg);
              F.removeParamAttr(i, llvm::Attribute::ImmArg);
          }
          ++i;
      }
    }
  }

  bool runOnModule(Module &M) override {
    removeImmarg(M);

    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeXOCCIRDowngrader(PassRegistry &Registry);
}

INITIALIZE_PASS(XOCCIRDowngrader, "xoccIRDowngrader",
  "pass that downgrades modern LLVM IR to something compatible with current xocc"
  "backend LLVM IR", false, false)
ModulePass *llvm::createXOCCIRDowngraderPass() {return new XOCCIRDowngrader();}

char XOCCIRDowngrader::ID = 0;
