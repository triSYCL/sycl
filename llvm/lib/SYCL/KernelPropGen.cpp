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
#include "llvm/IR/Attributes.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "SYCLUtils.h"

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
struct KernelPropGenState {

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
    if (Bundle->isDefaultBundle())
      return;
    LLVMContext &C = F.getContext();
    // Up to 2021.2 m_axi bundles were encoded with a call to sideeffect
    auto *BundleIDConstant =
        ConstantDataArray::getString(C, Bundle->BundleName, false);
    auto *MinusOne64 = ConstantInt::getSigned(IntegerType::get(C, 64), -1);
    auto *Zero32 = ConstantInt::getSigned(IntegerType::get(C, 32), 0);
    auto *CAZ =
        ConstantAggregateZero::get(ArrayType::get(IntegerType::get(C, 8), 0));
    auto *Slave = ConstantDataArray::getString(C, "slave", false);
    Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    SideEffect->addFnAttr(Attribute::NoUnwind);
    SideEffect->addFnAttr(Attribute::InaccessibleMemOnly);
    // TODO find a clever default value, allow user customization via
    // properties
    SideEffect->addFnAttr("xlx.port.bitwidth", "4096");

    OperandBundleDef OpBundle(
        "xlx_m_axi",
        ArrayRef<Value *>{&Arg, BundleIDConstant, MinusOne64, Slave, CAZ,
                          MinusOne64, MinusOne64, MinusOne64, MinusOne64,
                          MinusOne64, MinusOne64, Zero32});
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

  struct PipeEndpoint {
    /// Name of the kernel function in IR.
    StringRef Kernel;
    /// Name of the pipe function argument in IR.
    StringRef Arg;
  };
  struct PipeProp {
    /// Depth it defaults to -1 to indicate it is unset
    int Depth = -1;
    PipeEndpoint write;
    PipeEndpoint read;
  };

  /// Pipes are matched in read and write pairs by their ID. Their ID is a
  /// string matching the name of the global variable Intel uses to identify its
  /// pipes
  StringMap<PipeProp> PipeConnections;

  void collectPipeConnections(Module &M) {
    for (auto &F : M.functions())
      if (sycl::isKernelFunc(&F))
        for (auto &Arg : F.args())
          if (sycl::isPipe(&Arg)) {
            /// Build the endpoint for this pipe
            PipeEndpoint endPoint{F.getName(), Arg.getName()};
            PipeProp &Prop = PipeConnections[sycl::getPipeID(&Arg)];

            /// Figure out the correct endpoint to write to
            PipeEndpoint &mapEndPoint =
                sycl::isReadPipe(&Arg) ? Prop.read : Prop.write;
            assert(mapEndPoint.Arg.empty() && mapEndPoint.Kernel.empty() &&
                   "multiple reader or writers");
            mapEndPoint = endPoint;

            /// If the Depth is unset, set it
            if (Prop.Depth == -1)
              Prop.Depth = sycl::getPipeDepth(&Arg);

            assert(sycl::getPipeDepth(&Arg) == Prop.Depth &&
                   "read and write depth not matching");
          }
  }

  /// Print in O the property file for all kernels of M
  void generateProperties(Module &M, llvm::raw_fd_ostream &O) {
    json::OStream J(O, 2);
    llvm::json::Array Kernels{};
    bool SyclHlsFlow = Triple(M.getTargetTriple()).isXilinxHLS();
    bool VitisHlsFlow = Triple(M.getTargetTriple()).getArch() == llvm::Triple::vitis_ip;

    collectPipeConnections(M);

    J.objectBegin();
    J.attributeBegin("pipe_connections");
    J.arrayBegin();
    for (auto& Elem : PipeConnections) {
      J.objectBegin();
      J.attribute("writer_kernel", Elem.second.write.Kernel);
      J.attribute("writer_arg", Elem.second.write.Arg);
      J.attribute("reader_kernel", Elem.second.read.Kernel);
      J.attribute("reader_arg", Elem.second.read.Arg);
      J.attribute("depth", Elem.second.Depth);
      J.objectEnd();
    }
    J.arrayEnd();
    J.attributeEnd();
    J.attributeBegin("kernels");
    J.arrayBegin();
    for (auto &F : M.functions()) {
      if (sycl::isKernelFunc(&F)) {
        KernelProperties KProp(F, SyclHlsFlow);
        J.objectBegin();
        J.attribute("name", F.getName());
        auto ExtraArgs = getExtraArgs(F);
        if (ExtraArgs)
          J.attribute("extra_args", ExtraArgs.getValue());
        J.attributeBegin("bundle_hw_mapping");
        J.arrayBegin();
        for (auto &Bundle : KProp.getMAXIBundles()) {
          J.objectBegin();
          J.attribute("maxi_bundle_name", Bundle.BundleName);
          if (Bundle.TargetId.hasValue()) {
            StringRef Prefix;
            switch (Bundle.MemType) {
              case KernelProperties::MemoryType::ddr:
              Prefix = "DDR";
              break;
              case KernelProperties::MemoryType::hbm:
              Prefix = "HBM";
              break;
              default:
              llvm_unreachable("Default bundle should not appear here");
            }
            J.attribute("target_bank", formatv("{0}[{1}]", Prefix, Bundle.TargetId.getValue()));
          }
          J.objectEnd();
        }
        J.arrayEnd();
        J.attributeEnd();
        J.attributeBegin("arg_bundle_mapping");
        J.arrayBegin();
        for (auto &Arg : F.args()) {
          if (VitisHlsFlow)
            continue;
          /// Vitis's clang doesn't support string attributes on arguments which
          /// we use to annotate a pipe, so we remove it here. But we could
          /// remove them in the downgrader instead too.
          if (sycl::isPipe(&Arg))
            sycl::removePipeAnnotation(&Arg);
          else if (KernelProperties::isArgBuffer(&Arg, SyclHlsFlow)) {
            // This currently forces a default assignment of DDR banks to 0
            // as some platforms have different Default DDR banks and buffers
            // default to DDR Bank 0. Perhaps it is possible to query the
            // specific platform and reassign the buffers to different default
            // DDR banks based on the platform. But this would require a
            // change for every new platform. In either case, this puts in
            // infrastructure to assign DDR banks at compile time for a CU
            // if the information is passed down.
            const auto *Bundle = KProp.getArgumentMAXIBundle(&Arg);
            assert(Bundle && "Empty bundle should be marked as default bundle");
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
  bool runOnModule(Module &M) {
    llvm::raw_fd_ostream O(getWriteStreamId(KernelPropGenOutput),
                           true /*close in destructor*/);

    if (O.has_error())
      return false;
    generateProperties(M, O);

    // The module probably changed
    return true;
  }
};

void runKernelPropGen(Module &M) {
  KernelPropGenState S;
  S.runOnModule(M);
}

} // namespace

PreservedAnalyses KernelPropGenPass::run(Module &M, ModuleAnalysisManager &AM) {
  runKernelPropGen(M);
  return PreservedAnalyses::none();
}

namespace llvm {
void initializeKernelPropGenLegacyPass(PassRegistry &Registry);
} // namespace llvm

struct KernelPropGenLegacy : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  KernelPropGenLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    runKernelPropGen(M);
    return true;
  }
};

INITIALIZE_PASS(KernelPropGenLegacy, "kernelPropGen",
                "pass that finds kernel names and places them into a text file",
                false, false)
ModulePass *llvm::createKernelPropGenLegacyPass() { return new KernelPropGenLegacy(); }

char KernelPropGenLegacy::ID = 0;
