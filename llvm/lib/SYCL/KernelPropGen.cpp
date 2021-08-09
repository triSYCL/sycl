//===- KernelPropGen.cpp ---------------===//
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

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCL/KernelPropGen.h"
#include "llvm/SYCL/KernelProperties.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> KernelPropGenOutput("sycl-kernel-propgen-output",
                                                cl::ReallyHidden);

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

/// Retrieve the names and properties for all kernels in the module and place
/// them into a file. Generates the vitis HLS IR for kernel interface control.
struct KernelPropGen : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  KernelPropGen() : ModulePass(ID) {}

  /// Test if a function is a SPIR kernel
  bool isKernel(const Function &F) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL ||
        F.hasFnAttribute("fpga.top.func"))
      return true;
    return false;
  }

  int getWriteStreamId(StringRef Path) {
    int FileFD = 0;
    std::error_code EC = llvm::sys::fs::openFileForWrite(Path, FileFD);
    if (EC) {
      llvm::errs() << "Error in KernelPropGen Pass: " << EC.message() << "\n";
    }

    return FileFD;
  }

  static StringRef kindOf(const char *Str) {
    return StringRef(Str, strlen(Str) + 1);
  }

  /// Insert calls to sideeffect that will instruct Vitis HLS to put
  /// Arg in the bundle Bundle
  void generateBundleSE(Argument &Arg,
                        KernelProperties::MAXIBundle const *Bundle, Function &F,
                        Module &M) {
    LLVMContext &C = F.getContext();
    auto *BundleIDConstant =
        ConstantDataArray::getString(C, Bundle->BundleName, false);
    auto *MinusOne = ConstantInt::getSigned(IntegerType::get(C, 64), -1);
    auto *CAZ =
        ConstantAggregateZero::get(ArrayType::get(IntegerType::get(C, 8), 0));
    Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    SideEffect->addFnAttr(Attribute::NoUnwind);
    SideEffect->addFnAttr(Attribute::InaccessibleMemOnly);
    // TODO find a clever default value, allow user customization via properties
    SideEffect->addFnAttr("xlx.port.bitwidth", "4096");

    OperandBundleDef OpBundle(
        "xlx_m_axi",
        ArrayRef<Value *>{&Arg, BundleIDConstant, MinusOne, CAZ, CAZ, MinusOne,
                          MinusOne, MinusOne, MinusOne, MinusOne, MinusOne});
    Instruction *Instr = CallInst::Create(SideEffect, {}, {OpBundle});
    Instr->insertBefore(F.getEntryBlock().getTerminator());
  }

  Optional<std::string> getExtraArgs(Function &F) {
    if (F.hasFnAttribute("fpga.vpp.extraargs")) {
      return std::string{
          F.getFnAttribute("fpga.vpp.extraargs").getValueAsString()};
    }
    return {};
  }

  /// Print in O the property file for all kernels of M
  void generateProperties(Module &M, llvm::raw_fd_ostream &O) {
    json::OStream J(O, 2);
    llvm::json::Array Kernels{};
    bool SyclHlsFlow = Triple(M.getTargetTriple()).isXilinxHLS();

    J.objectBegin();
    J.attributeBegin("kernels");
    J.arrayBegin();
    for (auto &F : M.functions()) {
      if (isKernel(F)) {
        KernelProperties KProp(F, SyclHlsFlow);
        J.objectBegin();
        J.attribute("name", F.getName());
        auto extraArgs = getExtraArgs(F);
        if (extraArgs)
            J.attribute("extra_args", extraArgs.getValue());
        J.attributeBegin("bundle_hw_mapping");
        J.arrayBegin();
        for (auto &Bundle : KProp.getMAXIBundles()) {
          J.objectBegin();
          J.attribute("maxi_bundle_name", Bundle.BundleName);
          J.attribute("target_bank", formatv("DDR[{0}]", Bundle.TargetId));
          J.objectEnd();
        }
        J.arrayEnd();
        J.attributeEnd();
        J.attributeBegin("arg_bundle_mapping");
        J.arrayBegin();
        for (auto &Arg : F.args()) {
          if (KernelProperties::isArgBuffer(&Arg, SyclHlsFlow)) {
            // This currently forces a default assignment of DDR banks to 0
            // as some platforms have different Default DDR banks and buffers
            // default to DDR Bank 0. Perhaps it is possible to query the
            // specific platform and reassign the buffers to different default
            // DDR banks based on the platform. But this would require a
            // change for every new platform. In either case, this puts in
            // infrastructure to assign DDR banks at compile time for a CU
            // if the information is passed down.
            const auto *Bundle = KProp.getArgumentMAXIBundle(&Arg);
            assert(Bundle && "Empty bundle should default to DDR bank 0");
            generateBundleSE(Arg, Bundle, F, M);
            J.objectBegin();
            J.attribute("arg_name", Arg.getName());
            J.attribute("maxi_bundle_name", Bundle->BundleName);
            J.objectEnd();
          }
        }
        J.arrayEnd();
        J.attributeEnd();
        J.objectEnd();
      }
    }
    J.arrayEnd();
    J.attributeEnd();
    J.objectEnd();
  }

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    llvm::raw_fd_ostream O(getWriteStreamId(KernelPropGenOutput),
                           true /*close in destructor*/);

    if (O.has_error())
      return false;
    generateProperties(M, O);

    // The module probably changed
    return true;
  }
};

} // namespace

namespace llvm {
void initializeKernelPropGenPass(PassRegistry &Registry);
} // namespace llvm

INITIALIZE_PASS(KernelPropGen, "kernelPropGen",
                "pass that finds kernel names and places them into a text file",
                false, false)
ModulePass *llvm::createKernelPropGenPass() { return new KernelPropGen(); }

char KernelPropGen::ID = 0;
