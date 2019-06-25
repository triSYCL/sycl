//===- ChessMassage.cpp                                      ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Erases and modifies IR incompatabilities with chess-clang backend
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/SYCL/ChessMassage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// IR conversion/downgrader pass for ACAP chess-clang
struct ChessMassage : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  ChessMassage() : ModulePass(ID) {}

  /// Removes immarg (immutable arg) bitcode attribute that is applied to
  /// function parameters. It was added in LLVM-9 (D57825), so as xocc catches
  /// up it can be removed
  /// Note: If you llvm-dis the output Opt .bc file with an LLVM that has the
  /// ImmArg attribute, it will reapply all of the ImmArg attributes to the
  /// LLVM IR
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

  /// Removes SPIR_FUNC/SPIR_KERNEL calling conventions from functions and
  /// replace them with the default C calling convention for now
  void modifySPIRCallingConv(Module &M) {
    for (auto &F : M.functions()) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL ||
          F.getCallingConv() == CallingConv::SPIR_FUNC) {
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
        for (auto U : F.users()) {
          if (auto CI = dyn_cast<CallInst>(U))
            CI->setCallingConv(CallingConv::C);

          if (auto II = dyn_cast<InvokeInst>(U))
            II->setCallingConv(CallingConv::C);
         }
      }
    }
  }

  /// Remove a piece of metadata we don't want
  void removeMetadata(Module &M, StringRef MetadataName) {
    llvm::NamedMDNode *Old =
      M.getOrInsertNamedMetadata(MetadataName);
    if (Old)
      M.eraseNamedMetadata(Old);
  }

  bool runOnModule(Module &M) override {
    removeImmarg(M);
    modifySPIRCallingConv(M);
    // This causes some problems with Tale when we generate a .sfg from a kernel
    // that contains this piece of IR, perhaps it's fine not to delete it
    // provided it's not empty. But at least for the moment it's empty and Tale
    // doesn't know how to handle it.
    removeMetadata(M, "llvm.linker.options");

    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeChessMassage(PassRegistry &Registry);
}

INITIALIZE_PASS(ChessMassage, "ChessMassage",
  "pass that downgrades modern LLVM IR to something compatible with current "
  "chess-clang backend LLVM IR", false, false)
ModulePass *llvm::createChessMassagePass() {return new ChessMassage();}

char ChessMassage::ID = 0;
