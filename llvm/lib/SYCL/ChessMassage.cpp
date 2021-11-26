//===- ChessMassage.cpp ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// resolve IR incompatibilities with the chess backend.
// for the kernel merging process this pass also reorder function in the module,
// generate an ordered list of kernels and mark redundant kernel private.
// for more detail about kernel merging look at sycl-chess comment.
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/SYCL/ChessMassage.h"
#include "llvm/Transforms/Utils/FunctionComparator.h"

#include "DownGradeUtils.h"

#define DEBUG_TYPE "chess-massage"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

cl::opt<std::string> KernelUnmergedProperties("sycl-kernel-unmerged-prop-out",
                                              cl::ReallyHidden);

/// IR conversion/downgrader pass for ACAP chess-clang
struct ChessMassage : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  ChessMassage() : ModulePass(ID) {}

  GlobalNumberState GNS;

  /// For the kernel merging process is kernel needs to be treated specially.
  /// The function merger pass doesn't merge function with different arguments
  /// that it doesn't know about. so preparechess adds a unique attribute to
  /// every kernels to prevent kernel function from being merged by the function
  /// merger. This function cleans up this annotations. the function merger
  /// should not be run after this pass.
  void removeMetadataForUnmergability(Module &M) {
    for (auto &F : M.functions())
      if (F.hasFnAttribute("unmergable-kernel-id"))
        F.removeFnAttr("unmergable-kernel-id");
  }

  // Re-order functions according to their relative order in FunctionComparator
  // values. This means, the original
  // list is sorted as:
  //   func1 - func2 - func3 - func4 - func5 - ...
  // The merged one looks like:
  //   func1 - func4 - ...
  // And this indicates that func2 and func3 are merged into func1
  void TriageKernelForMerging(Module &M, llvm::raw_fd_ostream &O) {
    std::vector<Function *> Funcs;
    DenseMap<std::pair<Function*, Function*>, int> FuncCompareCache;

    /// Return the result of the comparaison of 2 function. with some caching.
    auto CompareFunc = [&](Function *LHS, Function *RHS) {
      auto Lookup = FuncCompareCache.find({LHS, RHS});
      int Compare = 0;
      if (Lookup != FuncCompareCache.end())
        /// We have a result in cache return it.
        Compare = Lookup->second;
      else {
        /// We do not have the result in cache. calculate it and write it to
        /// cache.
        Compare = FunctionComparator(LHS, RHS, &GNS).compare();
        LLVM_DEBUG(llvm::dbgs()
                   << "chess-massage: " << LHS->getName() << " <=> "
                   << RHS->getName() << " == " << Compare << "\n");
        FuncCompareCache[{LHS, RHS}] = Compare;
        FuncCompareCache[{RHS, LHS}] = -Compare;
      }
      return Compare;
    };
    llvm::SmallString<512> kernelNames;

    for (auto &F : M.functions()) {
      // Collect the kernel functions
      if (F.hasFnAttribute("chess_sycl_kernel"))
        Funcs.emplace_back(&F);
    }

    // Sort collected kernel functions by comparaison.
    llvm::sort(Funcs, [&](Function *LHS, Function *RHS) {
      return CompareFunc(LHS, RHS) < 0;
    });

    // Put sorted kernel functions back to the Modules's function list
    // Set linkages such that only the first of each function comparing equal
    // will survive globaldce
    for (auto I = Funcs.begin(), IE = Funcs.end(); I != IE; ++I) {
      Function *F = *I;
      F->removeFromParent();
      M.getFunctionList().push_back(F);
      if (I != Funcs.begin() && CompareFunc(*std::prev(I), F) == 0)
        /// This kenrel will be removed because it is redundant.
        F->setLinkage(llvm::GlobalValue::PrivateLinkage);
      else
        /// This kernel will be kept
        F->setLinkage(llvm::GlobalValue::ExternalLinkage);
      kernelNames += (" \"" + F->getName() + "\" \n").str();
      F->replaceAllUsesWith(UndefValue::get(F->getType()));
    }

    // Output our list of kernel names as a bash array we can iterate over
    if (!kernelNames.empty()) {
      O << "# ordered array of unmerged kernel names found in the current "
           "module\n";
      O << "declare -a KERNEL_NAME_ARRAY_UNMERGED=(" << kernelNames.str()
        << ")\n\n";
    }
  }

  /// Removes immarg (immutable arg) bitcode attribute that is applied to
  /// function parameters. It was added in LLVM-9 (D57825), so as Vitis catches
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

  int GetWriteStreamID(StringRef Path) {
    int FileFD = 0;
    std::error_code EC =
          llvm::sys::fs::openFileForWrite(Path, FileFD);
    if (EC) {
      llvm::errs() << "Error in KernelPropGen Pass: " << EC.message() << "\n";
    }

    return FileFD;
  }

  bool runOnModule(Module &M) override {
    llvm::raw_fd_ostream O(GetWriteStreamID(KernelUnmergedProperties),
                           true /*close in destructor*/);

    // Script header/comment
    O << "# This is a generated bash script to inject environment information\n"
         "# containing kernel properties that we need so we can compile.\n"
         "# This script is called from sycl-chess scripts.\n";

    if (O.has_error())
      return false;

    removeMetadataForUnmergability(M);
    TriageKernelForMerging(M, O);
    removeImmarg(M);

    // This causes some problems with Tale when we generate a .sfg from a kernel
    // that contains this piece of IR, perhaps it's fine not to delete it
    // provided it's not empty. But at least for the moment it's empty and Tale
    // doesn't know how to handle it.
    llvm::removeMetadata(M, "llvm.linker.options");

    llvm::removeAttributes(
        M, {Attribute::MustProgress, Attribute::ByVal, Attribute::StructRet});

    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeChessMassagePass(PassRegistry &Registry);
}

/// TODO: split this pass into the kernel mergin part and the downgrading part.
INITIALIZE_PASS(ChessMassage, "ChessMassage",
  "pass that downgrades modern LLVM IR to something compatible with current "
  "chess-clang backend LLVM IR", false, false)
ModulePass *llvm::createChessMassagePass() {return new ChessMassage();}

char ChessMassage::ID = 0;
