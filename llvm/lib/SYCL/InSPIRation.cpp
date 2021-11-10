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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/SYCL/InSPIRation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {
// avoid recreation of regex's we won't alter at runtime

struct Prefix {
  std::string Str;
  std::regex Matcher;
};

// matches number between Z and _ (\d+)(?=_)
static const std::regex matchZVal{R"((\d+)(?=_))"};

// matches reqd_work_group_size based on it's current template parameter list of
// 3 digits, doesn't care what the next adjoining type is or however many there
// are in this case. Technically the demangler enforces spacing between the
// commas but just in case it ever changes.
static const std::regex matchReqdWorkGroupSize{
    R"(cl::sycl::xilinx::reqd_work_group_size<\d+,\s?\d+,\s?\d+,)"};

// Just matches integers
static const std::regex matchSomeNaturalInteger{R"(\d+)"};

/// Transform the SYCL kernel functions into v++ SPIR-compatible kernels
struct InSPIRation : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  InSPIRation() : ModulePass(ID) {}

  /// This function currently works by checking for certain prefixes, and
  /// removing them from the mangled name, this currently is used for
  /// get_global_id etc. (as we forcefully prefix it with __spir_ocl_), and
  /// the math builtins which are prefixed with __spirv_ocl_. An example is:
  ///   Z24__spir_ocl_get_global_idj - > Z13get_global_idj

  /// Note: running this call on real SPIRV builtins is unlikely to yield a
  /// working SPIR builtin as they 1) May not be named the same/have a SPIR
  /// equivalent 2) Are not necessarily function calls, but possibly a magic
  /// variable like __spirv_BuiltInGlobalSize, something more complex would be
  /// required.
  void removePrefixFromMangling(Function &F, const std::regex Match,
                                const std::string Namespace) {
    const auto funcName = F.getName().str();

    auto regexName = std::regex_replace(funcName, Match, "");
    if (funcName != regexName) {
      std::smatch capture;
      if (std::regex_search(funcName, capture, matchZVal)) {
        auto zVal = std::stoi(capture[0]);

        // The poor man's mangling to a spir builtin, we know that the function
        // type itself is fine, we just need to work out the _Z mangling as spir
        // built-ins are not prefixed with __spirv_ocl_ or __spir_ocl_. All
        // _Z is in this case is the number of characters in the functions name
        // which we can work out by removing the number of characters in the
        // prefix e.g. __spirv_ocl_ (12 characters)/__spir_ocl_ (11 characters)
        // or from the original mangled names _Z value SPIR manglings for
        // reference:
        // https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-2.0-built-in-functions
        F.setName("_Z" + std::to_string(zVal - Namespace.size()) + regexName);
      }
    }
  }

  /// remap spirv builtins towards function present in vitis libspir.
  void remapBuiltin(Function *F) {
    /// according to the opencl 2.1 spec fmax_common and max are the same except
    /// on nans and infinity where fmax_common will output undefined value
    /// whereas fmax has rules to follow.
    /// because of this it is legal to replace a fmax_common by fmax.
    static std::pair<StringRef, StringRef> Mapping[] = {
        {"_Z11fmax_common", "_Z4fmax"},
        {"_Z3Dot", "_Z3dot"},
    };
    for (std::pair<StringRef, StringRef> Elem : Mapping)
      if (F->getName().startswith(Elem.first))
        return F->setName(Elem.second +
                          F->getName().drop_front(Elem.first.size()));
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
  getReqdWorkGroupSize(const std::string &demangledName, LLVMContext &Ctx) {
    SmallVector<llvm::Metadata *, 8> reqdWorkGroupSize;
    std::smatch capture;

    // If we're here we have captured at least one reqd_work_group_size
    // we only really care about the first application, because multiple
    // uses of this property on one kernel are invalid.
    if (std::regex_search(demangledName, capture, matchReqdWorkGroupSize)) {
      /// \todo: Enforce the use of a single reqd_work_group_size in the
      /// template interface in someway at compile time
      auto Int32Ty = llvm::Type::getInt32Ty(Ctx);
      std::string s = capture[0];
      std::sregex_token_iterator workGroupSizes{s.begin(), s.end(),
                                                matchSomeNaturalInteger};
      // only really care about the first 3 values, anymore and the
      // reqd_work_group_size interface is incorrect
      for (unsigned i = 0;
           i < 3 && workGroupSizes != std::sregex_token_iterator{};
           ++i, ++workGroupSizes) {
        reqdWorkGroupSize.push_back(llvm::ConstantAsMetadata::get(
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
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 0))};
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
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 2))};
    llvm::NamedMDNode *OCLVerMD =
        M.getOrInsertNamedMetadata("opencl.ocl.version");
    OCLVerMD->addOperand(llvm::MDNode::get(Ctx, OCLVerElts));
  }

  /// Remove extra SPIRV metadata for now, doesn't really crash v++ but its
  /// not required. Another method would just be to modify the SYCL Clang
  /// frontend to generate the actual SPIR/OCL metadata we need rather than
  /// always SPIRV/CL++ metadata
  void removeOldMetadata(Module &M) {
    llvm::NamedMDNode *Old = M.getOrInsertNamedMetadata("spirv.Source");
    if (Old)
      M.eraseNamedMetadata(Old);
  }

  /// Set the output Triple to SPIR
  void setSPIRTriple(Module &M) { M.setTargetTriple("spir64"); }

  /// Test if a function is a SPIR kernel
  bool isKernel(const Function &F) {
    return F.getCallingConv() == CallingConv::SPIR_KERNEL ||
           F.hasFnAttribute("fpga.top.func");
  }

  /// Test if a function is a non-intrinsic SPIR function, indicating that it is
  /// a user created function that the SYCL compiler has transitively generated
  /// or one that comes from an existing library of SPIR functions (HLS SPIR
  /// libraries)
  bool isTransitiveNonIntrinsicFunc(const Function &F) {
    if (!F.isIntrinsic())
      return true;
    return false;
  }

  /// This function gives llvm::function arguments with no name
  /// a default name e.g. arg_0, arg_1..
  ///
  /// This is because if your arguments have no name v++ will commit seppuku
  /// when generating XML. Perhaps it's possible to move this to the Clang
  /// Frontend by generating the name from the accessor/capture the arguments
  /// come from, but I believe it requires a special compiler invocation option
  /// to keep arg names from the frontend in the LLVM bitcode.
  void giveNameToArguments(Function &F) {
    int Counter = 0;
    for (auto &Arg : F.args()) {
      if (!Arg.hasName())
        Arg.setName("arg_" + Twine{Counter++});
    }
  }

  // Hopeful list/probably impractical asks for v++:
  // 1) Make XML generator/reader a little kinder towards arguments with no
  //   names if possible
  // 2) Allow -k all for LLVM IR input/SPIR-df so it can search for all
  //    SPIR_KERNEL's in a binary
  // 3) Be a little more name mangle friendly when reading in input e.g.
  //    accept: $_

  /// This pass should ideally be run after all your optimization passes,
  /// including anything aimed at fixing address spaces or simplifying
  /// load/stores. This is mainly as we would like to make the SSDM address
  /// space fixers job as simple as possible (if it gets overly complex or there
  /// needs to be some reorganization of passes detach it into a separate pass).
  ///
  /// However, it should be run prior to KernelPropGen as that
  /// pass relies on the kernel names generated here for now to fuel the driver
  /// script.
  bool runOnModule(Module &M) override {
    // funcCount is for naming new name for each function called in kernel
    int FuncCount = 0;

    std::vector<Function *> Declarations;
    for (auto &F : M.functions()) {
      if (isKernel(F)) {
        kernelSPIRify(F);
        applyKernelProperties(F);
        giveNameToArguments(F);

        /// \todo Possible: We don't modify declarations right now as this
        /// will destroy the names of SPIR/CL intrinsics as they aren't
        /// actually considered intrinsics by LLVM IR. If there is ever a need
        /// to modify declarations in someway then the best way to do it would
        /// be to have a comprehensive list of mangled SPIR intrinsic names
        /// and check against it. Note: This is only relevant if we still
        /// modify the name of every function to be sycl_func_x, if v++ ever
        /// gets a little friendlier to spir input, probably not required.
      } else if (isTransitiveNonIntrinsicFunc(F) && !F.isDeclaration()) {
        // After kernels code selection, there are only two kinds of functions
        // left: funcions called by kernels or LLVM intrinsic functions.
        // For functions called in SYCL kernels, put SPIR calling convention.
        kernelCallFuncSPIRify(F);

        // Modify the name of funcions called by SYCL kernel since function
        // names with $ sign would choke Xilinx v++.
        // And in Xilinx v++, there are passes splitting a function to new
        // functions. These new function names will come from some of the
        // basic block names in the original function.
        // So function and basic block names need to be modified to avoid
        // containing $ sign

        // Rename function name
        F.addFnAttr("src_name", F.getName());
        F.setName("sycl_func_" + Twine{FuncCount++});

        // While functions do come "named" it's in the form %0, %1 and v++
        // doesn't like this for the moment. v++ demands function arguments
        // be either unnamed or named non-numerically. This is a separate
        // issue from the reason we name kernel arguments (which is more
        // related to HLS needing names to generate XML).
        //
        // It doesn't require application to the SPIR intrinsics as we're
        // linking against the HLS SPIR library, which is already conformant.
        giveNameToArguments(F);
      } else if (isTransitiveNonIntrinsicFunc(F) && F.isDeclaration()) {
        // push back intrinsics to make sure we handle naming after changing the
        // name of all functions to sycl_func.
        // Note: if we do not rename all the functions to sycl_func_N, a more
        // complex modification to this pass may be required that makes sure all
        // functions on the device with the same name as a built-in are changed
        // so they have no conflict with the built-in functions.
        Declarations.push_back(&F);
      }
    }

    static Prefix prefix[] = {
        {"__spirv_ocl_u_", std::regex(R"((_Z\d+__spirv_ocl_u_))")},
        {"__spirv_ocl_s_", std::regex(R"((_Z\d+__spirv_ocl_s_))")},
        {"__spirv_ocl_", std::regex(R"((_Z\d+__spirv_ocl_))")},
        {"__spir_ocl_", std::regex(R"((_Z\d+__spir_ocl_))")},
        {"__spirv_", std::regex(R"((_Z\d+__spirv_))")},
        {"__spir_", std::regex(R"((_Z\d+__spir_))")},
    };

    for (auto F : Declarations) {
      // aims to catch things preceded by a namespace of the style:
      // _Z16__spirv_ocl_ and use the end section as a SPIR call
      // _Z24__spir_ocl_
      // _Z18__spirv_ocl_u_
      // _Z18__spirv_ocl_s_

      // This seems like a lazy brute force way to do things.
      // Perhaps there is a more elegant solution that can be implemented in the
      // future. I don't believe too much effort should be put into this until
      // the builtin implementation stabilizes
      for (Prefix p : prefix)
        removePrefixFromMangling(*F, p.Matcher, p.Str);

      remapBuiltin(F);
    }

    removeOldMetadata(M);
    if (!Triple(M.getTargetTriple()).isXilinxHLS()) {

      setSPIRVersion(M);

      setOpenCLVersion(M);

      // setSPIRTriple(M);

      /// TODO: Set appropriate layout so the linker doesn't always complain,
      /// this change may be better/more easily applied as something in the
      /// Frontend as we'd be lying about the layout if we didn't enforce it
      /// accurately in this pass. Which is potentially a good way to come
      /// across some weird runtime bugs.
      // setSPIRLayout(M);
    }

    // The module probably changed
    return true;
  }
};

} // namespace

namespace llvm {
void initializeInSPIRationPass(PassRegistry &Registry);
}

INITIALIZE_PASS(InSPIRation, "inSPIRation",
                "pass to make functions and kernels SPIR-compatible", false,
                false)
ModulePass *llvm::createInSPIRationPass() { return new InSPIRation(); }

char InSPIRation::ID = 0;
