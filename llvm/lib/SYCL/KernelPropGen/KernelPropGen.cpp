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
struct KernelPropGen : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  KernelPropGen() : ModulePass(ID) {}

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

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    SmallString<256> TDir;
    llvm::sys::path::system_temp_directory(true, TDir);
    std::string file = "KernelProperties_" + M.getSourceFileName();
    llvm::sys::path::append(TDir, file);
    llvm::sys::path::replace_extension(TDir, "sh",
      llvm::sys::path::Style::native);

    llvm::raw_fd_ostream O(GetWriteStreamID(TDir.str()),
                            true /*close in destructor*/);

    if (O.has_error())
      return false;
  
    // Generate a bash script to inject environment information containing 
    // kernel properties that we need so we can compile. This script is called
    // from sycl-xocc. #!/bin/bash
    O << "#!/bin/bash\n";
   
    std::string kernelNames = "";
    std::string DDRArgs = "";
    for (auto &F : M.functions()) {     
      if (isKernel(F)) {
        kernelNames += (" \"" + F.getName() + "\" ").str();
        
        for (auto& Arg : F.args()) {
          if (Arg.getType()->isPointerTy())
          // if the argument is a pointer in the global (1) or constant (3) 
          // address space it should be assigned to an explicit default DDR 
          // Bank of 0 to prevent assignment to DDR banks that are not 0.
          // This is to prevent mismatches between the SYCL runtime when 
          // declaring OpenCL buffers and the precompiled kernel, XRT will 
          // error out if there is a misamtch. Only OpenCL global memory is 
          // assigned to a DDR bank, this includes constant as it's just 
          // read-only global memory. 
          // \todo When adding an explicit way for users to specify DDR banks 
          // from the SYCL runtime this should be modified as well as the buffer 
          // XRT extensions.
          if (Arg.getType()->isPointerTy()
           && (Arg.getType()->getPointerAddressSpace() == 1 
           || Arg.getType()->getPointerAddressSpace() == 3)) {
              // This currently forces a default assignment of DDR banks to 0 
              // as some platforms have different Default DDR banks and buffers 
              // default to DDR Bank 0. Perhaps it is possible to query the 
              // specific platform and reassign the buffers to different default 
              // DDR banks based on the platform. But this would require a 
              // change for every new platform. In either case, this puts in 
              // infrastructure to assign DDR banks at compile time for a CU 
              // if the information is passed down.
              // This: Assigns a Default 0 DDR bank to all initial compute 
              // unit's, the _1 postfix to the kernel name represents the 
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
    if (!kernelNames.empty())
       O << "declare -a KERNEL_NAME_ARRAY=(" << kernelNames << ")\n";
    
    // output our --sp args containing DDR assignments.
    // Should look something like: --sp kernelName_1.arg_0:DDR[0] --sp ...
    if (!DDRArgs.empty())
       O << "DDR_BANK_ARGS=\"" << DDRArgs << "\"\n"; 
      
       
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
