//===- InSPIRation.cpp                                      ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Rewrite the kernels and functions so that they are compatible with SPIR
// representation as described in "The SPIR Specification Version 2.0 -
// Provisional" from Khronos Group.
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/SYCL/InSPIRation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CallSite.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

// avoid recreation of regex's we won't alter at runtime
// matches spirv ocl namespace
static std::regex matchSPIRVOCL {"(_Z[0-9]+__spirv_ocl_)"};
// matches number between Z and _ (?<=Z)(\\d+)(?=_)
static std::regex matchZVal {"(\\d)(\\d+)(?=_)"};

/// Transform the SYCL kernel functions into SPIR-compatible kernels
struct InSPIRation : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  InSPIRation() : ModulePass(ID) {}

  // Welcome to the world of assumptions, this works assuming the spirv
  // namespace in the SYCL namespace remains the same and assuming that the
  // functions are the same name as the spir built-ins.
  void renameSPIRVIntrinsicToSPIR(Function &F) {
    auto func_name = F.getName().str();
    auto regex_name = std::regex_replace(func_name,
                                         matchSPIRVOCL,
                                         "");

    if (func_name != regex_name) {
      std::cmatch capture;
      if (std::regex_search(func_name.c_str(), capture, matchZVal)) {
        auto zVal = std::stoi(capture[0]);

       // The poor mans mangling to a spir builtin, we know that the function
       // type itself is fine, we just need to work out the _Z mangling as spir
       // built-ins don't sit inside a namespace. All _Z is in this case is the
       // number of characters in the functions name which we can work out by
       // removing the number of characters in __spirv_ocl_ (12) from the
       // original mangled names _Z value.
       // SPIR manglings for reference:
       // https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-2.0-built-in-functions
       F.setName("_Z" + std::to_string(zVal - 12) + regex_name);
      }
    }
  }

  bool doInitialization(Module &M) override {
    // LLVM_DEBUG(dbgs() << "Enter: " << M.getModuleIdentifier() << "\n\n");

    // Do not change the code
    return false;
  }


  bool doFinalization(Module &M) override {
    // LLVM_DEBUG(dbgs() << "Exit: " << M.getModuleIdentifier() << "\n\n");
    // Do not change the code
    return false;
  }

  /// Do transforms on a SPIR function called by a SPIR kernel
  void kernelCallFuncSPIRify(Function &F) {
    // no op at the moment
  }

  /// Do transforms on a SPIR Kernel
  void kernelSPIRify(Function &F) {
    // no op at the moment
  }

  /// Add metadata for the SPIR 2.0 version
  void setSPIRVersion(Module &M) {
    /* Get InSPIRation from SPIRTargetCodeGenInfo::emitTargetMD in
       tools/clang/lib/CodeGen/TargetInfo.cpp */
    auto &Ctx = M.getContext();
    auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
    // SPIR v2.0 s2.12 - The SPIR version used by the module is stored in the
    // opencl.spir.version named metadata.
    llvm::Metadata *SPIRVerElts[] = {
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 2)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 0))
    };
    M.getOrInsertNamedMetadata("opencl.spir.version")
      ->addOperand(llvm::MDNode::get(Ctx, SPIRVerElts));
  }


  /// Add metadata for the OpenCL 1.2 version
  void setOpenCLVersion(Module &M) {
    auto &Ctx = M.getContext();
    auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
    // SPIR v2.0 s2.13 - The OpenCL version used by the module is stored in the
    // opencl.ocl.version named metadata node.
    llvm::Metadata *OCLVerElts[] = {
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 1)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 2))
    };
    llvm::NamedMDNode *OCLVerMD =
      M.getOrInsertNamedMetadata("opencl.ocl.version");
    OCLVerMD->addOperand(llvm::MDNode::get(Ctx, OCLVerElts));
  }

  /// Remove extra SPIRV metadata for now, doesn't really crash XOCC but its
  /// not required. Another method would just be to modify the SYCL Clang
  /// frontend to generate the actual SPIR/OCL metadata we need rather than
  /// always SPIRV/CL++ metadata
  void removeOldMetadata(Module &M) {
    llvm::NamedMDNode *Old =
      M.getOrInsertNamedMetadata("spirv.Source");
    if (Old)
      M.eraseNamedMetadata(Old);
  }

  /// Set the output Triple to SPIR
  void setSPIRTriple(Module &M) {
    M.setTargetTriple("spir64");
  }

  bool isKernel(const Function &F) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      return true;
    return false;
  }

  bool isTransitiveNonIntrinsicFunc(const Function &F) {
    if (F.getCallingConv() == CallingConv::SPIR_FUNC
        && !F.isIntrinsic())
      return true;
    return false;
  }

  /// Hopeful list/probably impractical asks for XOCC:
  /// 1) Make XML generator/reader a little kinder towards arguments with no names if possible
  /// 2) Allow -k all for llvm-ir input/spir-df so it can search for all SPIR_KERNEL's in a binary
  /// 3) Be a little more name mangle friendly when reading in input e.g. accept: $_

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    // funcCount is for naming new name for each function called in kernel
    int funcCount = 0, kernelCount = 0, counter = 0;

    std::vector<Function*> declarations;

    for (auto &F : M.functions()) {
        if (isKernel(F)) {
          kernelSPIRify(F);

          // F.setName("sycl_kernel_" + Twine{kernelCount++});

          // if your arguments have no name xocc will commit sepuku when
          // generating xml, so adding names to anonymous captures.
          // Perhaps it's possible to move this to the Clang Frontend by
          // generating the name from the accessor/capture the arguments
          // come from.
          counter = 0;
          for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
                I != E; ++I) {
              I->setName("arg_" + Twine{counter++});
          }

          // Rename basic block name
          // counter = 0;
          // for (auto &B : F)
          //   B.setName("label_" + Twine{counter++});


        // \todo Possible: We don't modify declarations right now as this will
        // destroy the names of SPIR/CL intrinsics as they aren't actually
        // considered intrinsics by LLVM IR. If there is ever a need to modify
        // declarations in someway then the best way to do it would be to have a
        // comprehensive list of mangled SPIR intrinsic names and check against
        // it. Note: This is only relevant if we still modify the name of every
        // function to be sycl_func_x, if xocc ever gets a little friendlier to
        // spir input, probably not required.
        } else if (isTransitiveNonIntrinsicFunc(F)
                    && !F.isDeclaration()) {
          // After kernels code selection, there are only two kinds of functions
          // left: funcions called by kernels or LLVM intrinsic functions.
          // For functions called in SYCL kernels, put SPIR calling convention.
          kernelCallFuncSPIRify(F);

          counter = 0;
          for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
                I != E; ++I) {
              I->setName("arg_" + Twine{counter++});
          }

          // Modify the name of funcions called by SYCL kernel since function
          // names with $ sign would choke Xilinx xocc.
          // And in Xilinx xocc, there are passes splitting a function to new
          // functions. These new function names will come from some of the
          // basic block names in the original function.
          // So function and basic block names need to be modified to avoid
          // containing $ sign

          // Rename function name
          F.setName("sycl_func_" + Twine{funcCount++});

          // Rename basic block name
          // counter = 0;
          // for (auto &B : F)
          //   B.setName("label_" + Twine{counter++});
      } else if (isTransitiveNonIntrinsicFunc(F)
                  && F.isDeclaration()) {
        // push back intrinsics to make sure we handle naming after changing the
        // name of all functions to sycl_func.
        // Note: if we stop the renaming of all functions to sycl_func_N a more
        // complex modification to this pass may be required that makes sure all
        // functions on the device with the same name as a built-in are changed
        // so they have no conflicts with the built-in functions.  
        declarations.push_back(&F);
      }
    }

    for (auto F : declarations) {
      // aims to catch things preceded by a namespace of the style:
      // _Z16__spirv_ocl_ and use the end section as a SPIR call
      renameSPIRVIntrinsicToSPIR(*F);
    }

    setSPIRVersion(M);

    setOpenCLVersion(M);

    setSPIRTriple(M);

    //setSPIRLayout(M);
    removeOldMetadata(M);

    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeInSPIRationPass(PassRegistry &Registry);
}

INITIALIZE_PASS(InSPIRation, "inSPIRation",
  "pass to make functions and kernels SPIR-compatible", false, false)
ModulePass *llvm::createInSPIRationPass() { return new InSPIRation(); }

char InSPIRation::ID = 0;
