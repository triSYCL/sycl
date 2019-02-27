//===- KernelNameGen.cpp                                      ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Retrieves the names of the kernels inside of the passed in file and places
// them into a text file. Possible to merge this into another pass if
// required, as it's a fairly trivial pass on its own.
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/SYCL/KernelNameGen.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Retrieve the names for all kernels in the module and place them into a file
struct KernelNameGen : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  KernelNameGen() : ModulePass(ID) {}

  bool isKernel(const Function &F) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      return true;
    return false;
  }

  int GetWriteStreamID(StringRef Path) {
    int FileFD = 0;
    std::error_code EC =
          llvm::sys::fs::openFileForWrite(Path, FileFD);
    if (EC) {
      llvm::errs() << "Error in KernelNameGen Pass: " << EC.message() << "\n";
    }

    return FileFD;
  }

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    // funcCount is for naming new name for each function called in kernel
    int funcCount = 0, kernelCount = 0, counter = 0;

    SmallString<256> TDir;
    llvm::sys::path::system_temp_directory(true, TDir);
    std::string file = "KernelNames_" + M.getSourceFileName();
    llvm::sys::path::append(TDir, file);
    llvm::sys::path::replace_extension(TDir, "txt",
      llvm::sys::path::Style::native);

    llvm::raw_fd_ostream O(GetWriteStreamID(TDir.str()),
                            true /*close in destructor*/);

    if (O.has_error())
      return false;

    for (auto &F : M.functions()) {
      if (isKernel(F)) {
        O << F.getName() << "\n";
      }
    }

    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeKernelNameGenPass(PassRegistry &Registry);
}

INITIALIZE_PASS(KernelNameGen, "kernelNameGen",
  "pass that finds kernel names and places them into a text file", false, false)
ModulePass *llvm::createKernelNameGenPass() { return new KernelNameGen(); }

char KernelNameGen::ID = 0;
