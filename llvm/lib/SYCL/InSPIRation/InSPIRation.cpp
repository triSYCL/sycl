//===- InSPIRation.cpp                                      ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Rewrite the kernels and functions so that they are compatible with SPIR
/// representation as described in "The SPIR Specification Version 2.0 -
/// Provisional" from Khronos Group.
///
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

/// \todo: Perhaps BOOST should appropriately be included through cmake. At the
/// moment it's found via existing in the environment I think.. seems a little
/// unclear at least. This goes for the run-time as well.
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
// avoid recreation of regex's we won't alter at runtime

// matches __spirv_ocl_ which is the transformed namespace of certain builtins
// in the cl::__spirv namespace after translation by the reflower (e.g. math
// functions like sqrt)
static const std::regex matchSPIRVOCL {R"((_Z\d+__spirv_ocl_))"};

// matches number between Z and _ (\d+)(?=_)
static const std::regex matchZVal {R"((\d+)(?=_))"};

// matches reqd_work_group_size based on it's current template parameter list of
// 3 digits, doesn't care what the next adjoining type is or however many there
// are in this case. Technically the demangler enforces spacing between the
// commas but just in case it ever changes.
static const std::regex matchReqdWorkGroupSize {
    R"(cl::sycl::xilinx::reqd_work_group_size<\d+,\s?\d+,\s?\d+,)"};

// Just matches integers
static const std::regex matchSomeNaturalInteger {R"(\d+)"};

// This is to give clarity to why we negate a value from the Z mangle component
// rather than having a magical number, we have the size of the string we've
// removed from the mangling.
static const std::string SPIRVNamespace("__spirv_ocl_");

/// Transform the SYCL kernel functions into xocc SPIR-compatible kernels
struct InSPIRation : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  InSPIRation() : ModulePass(ID) {}

  /// This function works assuming the built-ins inside of the cl::__spirv
  /// namespace undergo the transformation in the reflower to use the mangling
  /// __spirv_ocl_ in place of the regular namespace mangling. It also works
  /// assuming that the function contained inside the cl::__spirv namespace are
  /// named the same as an OpenCL/SPIR built-in e.g. it's still named sqrt with
  /// a valid SPIR/OpenCL overload.
  void renameSPIRVIntrinsicToSPIR(Function &F) {
    const auto funcName = F.getName().str();
    auto regexName = std::regex_replace(funcName,
                                        matchSPIRVOCL,
                                        "");

    if (funcName != regexName) {
      std::smatch capture;
      if (std::regex_search(funcName, capture, matchZVal)) {
        auto zVal = std::stoi(capture[0]);

        // The poor man's mangling to a spir builtin, we know that the function
        // type itself is fine, we just need to work out the _Z mangling as spir
        // built-ins don't sit inside a namespace. All _Z is in this case is the
        // number of characters in the functions name which we can work out by
        // removing the number of characters in __spirv_ocl_ (12) from the
        // original mangled names _Z value.
        // SPIR manglings for reference:
        // https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-2.0-built-in-functions
        F.setName("_Z" + std::to_string(zVal - SPIRVNamespace.size())
                + regexName);
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

  /// Retrieves the ReqdWorkGroupSize values from a demangled function name
  /// using regex.
  SmallVector<llvm::Metadata *, 8>
  getReqdWorkGroupSize(const std::string& demangledName, LLVMContext &Ctx) {
    SmallVector<llvm::Metadata *, 8> reqdWorkGroupSize;
    std::smatch capture;

    // If we're here we have captured at least one reqd_work_group_size
    // we only really care about the first application, because multiple
    // uses of this property on one kernel are invalid.
    if (std::regex_search(demangledName, capture, matchReqdWorkGroupSize)) {
      /// \todo: Enforce the use of a single reqd_work_group_size in the template
      /// interface in someway at compile time
      auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
      std::string s = capture[0];
      std::sregex_token_iterator workGroupSizes{s.begin(), s.end(),
                                            matchSomeNaturalInteger};
      // only really care about the first 3 values, anymore and the
      // reqd_work_group_size interface is incorrect
      for (unsigned i = 0;
           i < 3 && workGroupSizes != std::sregex_token_iterator{};
           ++i, ++workGroupSizes) {
        reqdWorkGroupSize.push_back(
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(Int32Ty, std::stoi(*workGroupSizes))));
      }

      if (reqdWorkGroupSize.size() != 3)
        report_fatal_error("The reqd_work_group_size properties dimensions are "
                           "not equal to 3");
    }

    return reqdWorkGroupSize;
  }

  /// In SYCL, kernel names are defined by types and in our current
  /// implementation we wrap our SYCL kernel names with properties that are
  /// defined as template types. For example ReqdWorkGroupSize is defined as
  /// one of these when the kernel name is translated from type to kernel name
  /// the information is retained and we can retrieve it in this LLVM pass by
  /// using regex on it.
  /// This is something we can improve on in the future, but the concept works
  /// for the moment.
  void applyKernelProperties(Function &F) {
    auto &ctx = F.getContext();

    auto funcMangledName = F.getName().str();
    auto demangledName = llvm::demangle(funcMangledName);
    auto reqdWorkGroupSize = getReqdWorkGroupSize(demangledName, ctx);

    if (!reqdWorkGroupSize.empty())
      F.setMetadata("reqd_work_group_size",
                    llvm::MDNode::get(ctx, reqdWorkGroupSize));
  }

  /// Sets a unique name to a function which is currently computed from a SHA-1
  /// hash of the original name.
  ///
  /// This unique name is used for kernel names so that they can be passed to
  /// the xocc compiler without error and then recomputed and used in the run
  /// time (program_manager) to correctly retrieve the kernel from the binary.
  /// This is required as xocc doesn't like certain characters in mangled names
  /// and we need a name that can be used in the run-time and passed to the
  /// compiler.
  /// The hash is recomputed in the run-time from the kernel name found in the
  /// integrated header as we currently do not wish to alter the integrated
  /// header with an LLVM pass as it will take some alteration to the driver
  /// and header that are not set in stone yet.
  /// Perhaps in the future that may be the direction that is taken however.
  void setUniqueName(Function &F) {
    // can technically use our own "namespace" to generate the SHA-1 rather than
    // ns::dns, it works for now for testing purposes
    // Note: LLVM has SHA-1, but if we use LLVM SHA-1 we can't recreate it in the
    // run-time. Perhaps it can be utilized in another way to achieve similar
    // results though.
    boost::uuids::name_generator_latest gen{boost::uuids::ns::dns()};

    // long uuid example: 8e6761a3-f150-580f-bae8-7d8d86bfa552
    boost::uuids::uuid uDoc = gen(F.getName().str());

    // converted to a hash value example: 14050332600208107103
    boost::hash<boost::uuids::uuid> uuidHasher;
    std::size_t uuidHashValue = uuidHasher(uDoc);

    // The uuid on it's own is too long for xocc it has a 64 character limit for
    // the kernels name and the name of its compute unit. By default the compute
    // unit name is the kernel name with an _N, uuid's are over 32 chars long so
    // 32*2 + another few characters pushes it over the limit.
    /// \todo In the middle term change this to take the lowest bits of the SHA-1
    ///       uuid e.g. take the string, regex to remove the '-' then take the
    ///       max characters you can fit from the end (30-31~?). Echo the changes
    ///       to the SYCL run-time's program_manager.cpp so the modified kernel
    ///       names are correctly computed and can be found in the binary.
    /// \todo In the long-term come up with a better way of doing this than
    ///       changing all the names to a SHA-1 hash. Like asking for an update
    ///       to the xocc compiler to accept characters that appear in mangled
    ///       names
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

  /// Remove extra SPIRV metadata for now, doesn't really crash xocc but its
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

  // Hopeful list/probably impractical asks for xocc:
  // 1) Make XML generator/reader a little kinder towards arguments with no
  //   names if possible
  // 2) Allow -k all for LLVM IR input/SPIR-df so it can search for all
  //    SPIR_KERNEL's in a binary
  // 3) Be a little more name mangle friendly when reading in input e.g.
  //    accept: $_

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    // funcCount is for naming new name for each function called in kernel
    int funcCount = 0;
    int counter = 0;

    std::vector<Function*> declarations;

    for (auto &F : M.functions()) {
        if (isKernel(F)) {
          kernelSPIRify(F);
          applyKernelProperties(F);
          setUniqueName(F);

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


        /// \todo Possible: We don't modify declarations right now as this will
        /// destroy the names of SPIR/CL intrinsics as they aren't actually
        /// considered intrinsics by LLVM IR. If there is ever a need to modify
        /// declarations in someway then the best way to do it would be to have a
        /// comprehensive list of mangled SPIR intrinsic names and check against
        /// it. Note: This is only relevant if we still modify the name of every
        /// function to be sycl_func_x, if xocc ever gets a little friendlier to
        /// spir input, probably not required.
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
        // Note: if we do not rename all the functions to sycl_func_N, a more
        // complex modification to this pass may be required that makes sure all
        // functions on the device with the same name as a built-in are changed
        // so they have no conflict with the built-in functions.
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
