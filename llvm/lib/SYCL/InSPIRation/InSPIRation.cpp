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
#include "llvm/Demangle/Demangle.h"
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

#define BOOST_NO_EXCEPTIONS

// TODO: Perhaps BOOST should appropriately be included through cmake. At the
// moment it's found via existing in the environment I think.. seems a little
// unclear at least. This goes for the run-time as well.
#include <boost/container_hash/hash.hpp> // uuid_hasher
#include <boost/uuid/uuid_generators.hpp> // sha name_gen/generator
#include <boost/uuid/uuid_io.hpp> // uuid to_string

// BOOST_NO_EXCEPTIONS enabled so we need to define our own throw_exception or
// get a linker error.
namespace boost {
  void throw_exception(std::exception const & e) {}
}

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

// Create static regex's to avoid recreation of regex's we won't alter at
// runtime

// matches spirv::ocl namespace AFTER reflower, this may change if the reflower
// gets removed.
static std::regex matchSPIRVOCL {"(_Z[0-9]+__spirv_ocl_)"};

// matches number between Z and _ (?<=Z)(\\d+)(?=_)
static std::regex matchZVal {"(\\d)(\\d+)(?=_)"};

// matches the reqd_work_group_size template's unmangled name and doesn't care
// what's in the angular brackets. So the same capture could work for any
// property.
static std::regex matchReqdWorkGroupSize {"cl::sycl::xilinx::reqd_work_group_size<(.*?)>"};

// Just matches integers
static std::regex matchInt {"[0-9]+"};

/// Transform the SYCL kernel functions into SPIR-compatible kernels
struct InSPIRation : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  InSPIRation() : ModulePass(ID) {}

  // Welcome to the world of assumptions, this works assuming the spirv
  // namespace in the SYCL namespace remains the same and assuming that the
  // functions are the same name as the spir built-ins.
  void renameSPIRVIntrinsicToSPIR(Function &F) {
    auto funcName = F.getName().str();
    auto regexName = std::regex_replace(funcName,
                                        matchSPIRVOCL,
                                        "");

    if (funcName != regexName) {
      std::cmatch capture;
      if (std::regex_search(funcName.c_str(), capture, matchZVal)) {
        auto zVal = std::stoi(capture[0]);

       // The poor mans mangling to a spir builtin, we know that the function
       // type itself is fine, we just need to work out the _Z mangling as spir
       // built-ins don't sit inside a namespace. All _Z is in this case is the
       // number of characters in the functions name which we can work out by
       // removing the number of characters in __spirv_ocl_ (12) from the
       // original mangled names _Z value.
       // SPIR manglings for reference:
       // https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-2.0-built-in-functions
       F.setName("_Z" + std::to_string(zVal - 12) + regexName);
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

  auto getReqdWorkGroupSize(std::string demangledName, LLVMContext &Ctx) {
    std::cmatch capture;

    SmallVector<llvm::Metadata *, 8> reqdWorkGroupSize;

    if (std::regex_search(demangledName.c_str(), capture, matchReqdWorkGroupSize)) {
      // if we're here we have captured at least one reqd_work_group_size
      // we only really care about the first application, because multiple
      // uses of this property on one kernel are invalid.
      // TODO: Enforce the use of a single reqd_work_group_size in the template
      // interface in someway at compile time
       std::string s = capture[0].str();
       std::sregex_token_iterator rend;
       std::sregex_token_iterator a ( s.begin(), s.end(), matchInt );

       // only really care about the first 3 values, anymore and the
       // reqd_work_group_size interface is incorrect
       unsigned i = 0;
       auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
       while (a!=rend && i < 3) {
         reqdWorkGroupSize.push_back(
             llvm::ConstantAsMetadata::get(
                 llvm::ConstantInt::get(Int32Ty, std::stoi(*a++))));
        ++i;
      }
    }

    return reqdWorkGroupSize;
  }

  /// Apply properties to a kernel
  void applyKernelProperties(Function &F) {
    auto &ctx = F.getContext();

    auto funcMangledName = F.getName().str();
    auto demangledName = demangle(funcMangledName);
    auto reqdWorkGroupSize = getReqdWorkGroupSize(demangledName, ctx);

    if (reqdWorkGroupSize.size() == 3)
      F.setMetadata("reqd_work_group_size",
                    llvm::MDNode::get(ctx, reqdWorkGroupSize));
  }

  /// Transform a mangled kernel name to a hash that can be given to xocc
  /// without error and used in the run time to correctly retrieve the kernel
  void hashKernelName(Function &F) {
    llvm::errs() << "function name mangling before hash converison: "
                 << F.getName().str() << "\n";
    // can technically use our own "namespace" to generate the sha1 rather than
    // ns::dns, it works for now for testing purposes
    // Note: LLVM has SHA1, but if we use LLVM sha1 we can't recreate it in the
    // run-time. Perhaps it can be utilized in another way to achieve similar
    // results though.
    boost::uuids::name_generator_sha1 gen(boost::uuids::ns::dns());

    // long uid example: 8e6761a3-f150-580f-bae8-7d8d86bfa552
    boost::uuids::uuid uDoc = gen(F.getName().str());
    llvm::errs() << "as uid: " << boost::uuids::to_string(uDoc) << "\n";

    // converted to a hash value example: 14050332600208107103
    boost::hash<boost::uuids::uuid> uuidHasher;
    std::size_t uuidHashValue = uuidHasher(uDoc);
    llvm::errs() << "converted to a hash value: " << uuidHashValue << "\n";

    // The uid on it's own is too long for xocc it has a 64 character limit for
    // the kernels name and the name of its compute unit. By default the compute
    // unit name is the kernel name with an _N, uid's are over 32 chars long so
    // 32*2 + another few characters pushes it over the limit. Perhaps its
    // possible to just take part of it but it may be harder to avoid name
    // collisions that way.
    F.setName(std::to_string(uuidHashValue));
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
    int funcCount = 0, counter = 0;

    std::vector<Function*> declarations;

    for (auto &F : M.functions()) {
        if (isKernel(F)) {
          kernelSPIRify(F);
          applyKernelProperties(F);
          hashKernelName(F);

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
