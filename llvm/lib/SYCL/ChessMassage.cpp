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

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/SYCL/ChessMassage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
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

  void removeMetadataForUnmergability(Module &M) {
    for (auto &F : M.functions())
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
        F.removeFnAttr("unmergable-kernel-id");
  }

  // Re-order functions with hash values. This is taken from MergeFunctions
  // and required to correctly identifie to which one the functions are
  // merged into. This means, the original list is sorted as:
  //   func1 - func2 - func3 - func4 - func5 - ...
  // The merged one looks like:
  //   func1 - func4 - ...
  // And this indicates that func2 and func3 are merged into func1
  // Note, since it's from MergeFunctions, if MergeFunctions changes in some
  // way, it may break, ex identifying merged function incorrectly.
  void prepareForMerging(Module &M, llvm::raw_fd_ostream &O) {
    std::vector<Function *> Funcs;
    DenseMap<Function*, FunctionComparator::FunctionHash> FuncHashCache;
    DenseMap<std::pair<Function*, Function*>, int> FuncCompareCache;
    auto ComapreFunc = [&](Function *LHS, Function *RHS) {
      auto Lookup = FuncCompareCache.find({LHS, RHS});
      int Compare = 0;
      if (Lookup != FuncCompareCache.end())
        Compare = Lookup->second;
      else {
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
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
        Funcs.emplace_back(&F);
    }

    // Sort collected kernel functions with hash values
    llvm::sort(Funcs, [&](Function *LHS, Function *RHS) {
      return ComapreFunc(LHS, RHS) < 0;
    });

    // Put sorted kernel functions back to the Modules's function list
    // Set linkages such that only one of each function comparing equal will survive globaldce
    for (auto I = Funcs.begin(), IE = Funcs.end(); I != IE; ++I) {
      Function *F = *I;
      F->removeFromParent();
      M.getFunctionList().push_back(F);
      if (I != Funcs.begin() && ComapreFunc(*std::prev(I), F) == 0)
        F->setLinkage(llvm::GlobalValue::PrivateLinkage);
      else
        F->setLinkage(llvm::GlobalValue::ExternalLinkage);
      kernelNames += (" \"" + F->getName() + "\" \n").str();
      F->replaceAllUsesWith(UndefValue::get(F->getType()));
    }

    // output our list of kernel names as a bash array we can iterate over
    if (!kernelNames.empty()) {
       O << "# array of unmerged kernel names found in the current module\n";
       O << "declare -a KERNEL_NAME_ARRAY_UNMERGED=(" << kernelNames.str()
         << ")\n\n";
    }
  }

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
        if (F.getCallingConv() == CallingConv::SPIR_KERNEL &&
            F.getLinkage() != llvm::GlobalValue::InternalLinkage)
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
        for (auto U : F.users()) {
          if (auto CB = dyn_cast<CallBase>(U))\
            CB->setCallingConv(CallingConv::C);
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
         "# This script is called from the sycl-xocc and sycl-chess scripts.\n";

    if (O.has_error())
      return false;

    removeMetadataForUnmergability(M);
    prepareForMerging(M, O);
    removeImmarg(M);
    // This has to be done before changing the calling convention
    modifySPIRCallingConv(M);
    // This causes some problems with Tale when we generate a .sfg from a kernel
    // that contains this piece of IR, perhaps it's fine not to delete it
    // provided it's not empty. But at least for the moment it's empty and Tale
    // doesn't know how to handle it.
    removeMetadata(M, "llvm.linker.options");

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

INITIALIZE_PASS(ChessMassage, "ChessMassage",
  "pass that downgrades modern LLVM IR to something compatible with current "
  "chess-clang backend LLVM IR", false, false)
ModulePass *llvm::createChessMassagePass() {return new ChessMassage();}

char ChessMassage::ID = 0;
