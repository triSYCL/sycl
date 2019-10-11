//===- KernelPropGen.cpp                                      ---------------===//
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

#include "llvm/SYCL/KernelPropGen.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Enum for Address Space values, used in ASFixer pass and LLVM-SPIRV
enum SPIRAddressSpace {
  SPIRAS_Private,  // Address space: 0
  SPIRAS_Global,   // Address space: 1
  SPIRAS_Constant, // Address space: 2
  SPIRAS_Local,    // Address space: 3
  SPIRAS_Generic,  // Address space: 4
};

/// Retrieve the names for all kernels in the module and place them into a file
struct KernelPropGen : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  KernelPropGen() : ModulePass(ID) {}

  /// Test if a function is a SPIR kernel
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
      llvm::errs() << "Error in KernelPropGen Pass: " << EC.message() << "\n";
    }

    return FileFD;
  }

  void GenerateXOCCPropertyScript(llvm::raw_fd_ostream &O, Module &M) {
    llvm::SmallString<512> kernelNames;
    llvm::SmallString<512> DDRArgs;
    for (auto &F : M.functions()) {
      if (isKernel(F)) {
        kernelNames += (" \"" + F.getName() + "\" ").str();

        for (auto& Arg : F.args()) {
          if (Arg.getType()->isPointerTy())
          // if the argument is a pointer in the global or constant
          // address space it should be assigned to an explicit default DDR
          // Bank of 0 to prevent assignment to DDR banks that are not 0.
          // This is to prevent mismatches between the SYCL runtime when
          // declaring OpenCL buffers and the pre-compiled kernel, XRT will
          // error out if there is a mismatch. Only OpenCL global memory is
          // assigned to a DDR bank, this includes constant as it's just
          // read-only global memory.
          // \todo When adding an explicit way for users to specify DDR banks
          // from the SYCL runtime this should be modified as well as the buffer
          // XRT extensions.
          if (Arg.getType()->isPointerTy()
              && (Arg.getType()->getPointerAddressSpace() == SPIRAS_Global
              || Arg.getType()->getPointerAddressSpace() == SPIRAS_Constant)) {
              // This currently forces a default assignment of DDR banks to 0
              // as some platforms have different Default DDR banks and buffers
              // default to DDR Bank 0. Perhaps it is possible to query the
              // specific platform and reassign the buffers to different default
              // DDR banks based on the platform. But this would require a
              // change for every new platform. In either case, this puts in
              // infrastructure to assign DDR banks at compile time for a CU
              // if the information is passed down.
              // This: Assigns a Default 0 DDR bank to all initial compute
              // unit's, the _1 post-fix to the kernel name represents the
              // default compute unit name. If more than one CU is generated
              // (which we don't support yet in any case) then they would be
              // KernelName_2..KernelName_3 etc.
              DDRArgs += ("--sp " + F.getName() + "_1." + Arg.getName()
                          + ":DDR[0] ").str();
          }
        }
        O << "\n"; // line break for new set of kernel properties
      }
    }

    // output our list of kernel names as a bash array we can iterate over
    if (!kernelNames.empty()) {
       O << "# array of kernel names found in the current module\n";
       O << "declare -a KERNEL_NAME_ARRAY=(" << kernelNames.str() << ")\n\n";
    }

    // output our --sp args containing DDR assignments.
    // Should look something like: --sp kernelName_1.arg_0:DDR[0] --sp ...
    if (!DDRArgs.empty()) {
       O <<   "# list of kernel arguments to ddr bank mappings for arguments\n"
              "# in the global and constant address spaces, passed to the\n"
              "# xocc linker phase\n";
       O << "DDR_BANK_ARGS=\"" << DDRArgs.str() << "\"\n";
    }
  }

  void GenerateChessPropertyScript(llvm::raw_fd_ostream &O, Module &M) {
    llvm::SmallString<512> kernelNames;
    for (auto &F : M.functions()) {
      if (isKernel(F)) {
        kernelNames += (" \"" + F.getName() + "\" ").str();
      }
    }

    // output our list of kernel names as a bash array we can iterate over
    if (!kernelNames.empty()) {
       O << "# array of kernel names found in the current module\n";
       O << "declare -a KERNEL_NAME_ARRAY=(" << kernelNames.str() << ")\n\n";
    }
  }

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    SmallString<256> TDir;
    llvm::sys::path::system_temp_directory(true, TDir);
    // Make sure to rip off the directories for the filename
    llvm::Twine file = "KernelProperties_" +
      llvm::sys::path::filename(M.getSourceFileName());
    llvm::sys::path::append(TDir, file);
    llvm::sys::path::replace_extension(TDir, "bash",
      llvm::sys::path::Style::native);
    llvm::raw_fd_ostream O(GetWriteStreamID(TDir.str()),
                            true /*close in destructor*/);

   // Script header/comment
    O << "# This is a generated bash script to inject environment information\n"
         "# containing kernel properties that we need so we can compile.\n"
         "# This script is called from the sycl-xocc and sycl-chess scripts.\n";

    if (O.has_error())
      return false;

    auto T = llvm::Triple(M.getTargetTriple());
    if (T.isXilinxAIE())
      GenerateChessPropertyScript(O, M);
    else
      GenerateXOCCPropertyScript(O, M);

    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeKernelPropGenPass(PassRegistry &Registry);
}

INITIALIZE_PASS(KernelPropGen, "kernelPropGen",
  "pass that finds kernel names and places them into a text file", false, false)
ModulePass *llvm::createKernelPropGenPass() { return new KernelPropGen(); }

char KernelPropGen::ID = 0;
