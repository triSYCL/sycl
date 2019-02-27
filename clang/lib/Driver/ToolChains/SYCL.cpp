//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCL.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;
using namespace llvm::sys;

///////////////////////////////////////////////////////////////////////////////
////                            XOCC Installation Detector
///////////////////////////////////////////////////////////////////////////////


XOCCInstallationDetector::XOCCInstallationDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args)
    : D(D) {
  // This might only work on Linux systems.
  // Rather than just checking the environment variables you could also add an
  // optional path variable for users to use.
  if (llvm::ErrorOr<std::string> xocc = findProgramByName("xocc")) {
      SmallString<256> xoccsAbsolutePath;
      fs::real_path(*xocc, xoccsAbsolutePath);

      BinaryPath = xoccsAbsolutePath.str();

      StringRef xoccDir = path::parent_path(xoccsAbsolutePath);

      if (path::filename(xoccDir) == "bin")
        BinPath = xoccDir;

      xoccDir = path::parent_path(xoccDir);
      if (path::filename(xoccDir) == "sdx") {
        SDXPath = xoccDir;
        LibPath = SDXPath + "/lnx64/lib";
      }
    // TODO: slightly stricter IsValid test.. check all strings aren't empty?
    IsValid = true;
  }
}

///////////////////////////////////////////////////////////////////////////////
////                            XOCC Assembler
///////////////////////////////////////////////////////////////////////////////

// NOTE: The Output argument is completely ignored in this Job, this Job does
// not currently output the .o file specified by the Output argument. It
// generates a number of temporary .xo files which are read in by the LinkerXOCC
// Tool which then links them together. Perhaps a better method needs to be
// designed, however it's difficult to work around XOCCs requirements outside of
// perhaps merging the XOCC Assembler and Linker stage into one.
void SYCL::AssemblerXOCC::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {

   const char *OptOutputName = constructOptCommand(C, JA, Inputs, Args,
                                                   Inputs[0].getFilename());

   const char *LLVMLinkOutputName = constructLLVMLinkCommand(C, JA,
                                                             Inputs, Args,
                                                             OptOutputName);

   const char *XOCCOutputName = constructXOCCCompileCommand(C, JA, Inputs, Args,
                                                            LLVMLinkOutputName);
}

// Inherited from HIP toolchain with some modifications.
const char *SYCL::AssemblerXOCC::constructOptCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, const char *InputFileName) const {
  // Construct opt command.
  ArgStringList CmdArgs;
  CmdArgs.push_back(InputFileName);

  // Pass optimization arg to opt.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    StringRef OOpt = "3";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // -Os, -Oz, and -O(anything else) map to -O2
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", "2")
                 .Case("z", "2")
                 .Default("2");
    }
    CmdArgs.push_back(Args.MakeArgString("-O" + OOpt));
  } else {
    CmdArgs.push_back(Args.MakeArgString("-O3"));
  }

  // output LLVM IR for debugging purposes
  // OptsArgs.push_back(Args.MakeArgString(-S"));

  // Desired Passes
  CmdArgs.push_back(Args.MakeArgString("-asfix"));
  CmdArgs.push_back(Args.MakeArgString("-globaldce"));
  CmdArgs.push_back(Args.MakeArgString("-inSPIRation"));
  CmdArgs.push_back(Args.MakeArgString("-globaldce"));

  // LLVM Pass that will generate the names for us, I don't want to
  // rely on a header that may change overtime and having to wire up
  // the phase dependency differently for XOCC seems like a waste
  CmdArgs.push_back(Args.MakeArgString("-kernelNameGen"));

  // Dispose of the file containing kernel names generated by KernelNameGen
  // after compilation is complete (KernelNames_sourcefilename.txt)
  // TODO: Move the creation of the KernelNameFile name and pass it into the
  // XOCC Compile Command Constructor and this constructor. Also probably make
  // it a little less complex...
  SmallString<256> KernelNameFile;
  path::system_temp_directory(true, KernelNameFile);
  path::append(KernelNameFile, "KernelNames_" +
                          std::string(Inputs[0].getBaseInput()));
  path::replace_extension(KernelNameFile, "txt",
                                     path::Style::native);
  C.addTempFile(C.getArgs().MakeArgString(KernelNameFile));

  CmdArgs.push_back("-o");

  // Just slightly modified bc output, but xocc won't take .bc named files..
  std::string TmpFileName = C.getDriver().GetTemporaryPath(
      std::string(Inputs[0].getBaseInput()) + "-optimized", "xpirbc");
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpFileName));
  CmdArgs.push_back(OutputFileName);
  SmallString<128> OptPath(C.getDriver().Dir);
  path::append(OptPath, "opt");
  const char *OptExec = Args.MakeArgString(OptPath);

  // NOTE: Perhaps not required to pass Inputs down, depending on what they are.
  C.addCommand(llvm::make_unique<Command>(JA, *this, OptExec, CmdArgs, Inputs));
  return OutputFileName;
}

const char *SYCL::AssemblerXOCC::constructLLVMLinkCommand(
  Compilation &C, const JobAction &JA,  const InputInfoList &Inputs,
  const llvm::opt::ArgList &Args, const char *InputFileName) const {

  const auto &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());

  ArgStringList CmdArgs;
  CmdArgs.push_back(InputFileName);

  // FIXME: Assuming a 64bit here
  // Library with our mangled CL intrinsics
  CmdArgs.push_back(C.getArgs().MakeArgString(TC.XOCCInstallation.getLibPath() +
                                              "/libspir64-39-hls.bc"));

  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  SmallString<128> TmpName(C.getDriver().GetTemporaryPath(
      std::string(Inputs[0].getBaseInput()) + "-linked", "xpirbc"));
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpName));
  CmdArgs.push_back(OutputFileName);
  SmallString<128> ExecPath(C.getDriver().Dir);
  path::append(ExecPath, "llvm-link");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, None));
  return OutputFileName;
}
const char *SYCL::AssemblerXOCC::constructXOCCCompileCommand(
  Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
  const llvm::opt::ArgList &Args, const char *InputFileName) const {

  const auto &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());

  ArgStringList CmdArgs;

  CmdArgs.push_back(
    C.getArgs().MakeArgString(TC.XOCCInstallation.getBinaryPath()));

  CmdArgs.push_back(C.getArgs().MakeArgString("-c"));

  // Long winded way to get the name of the file containing kernel names
  // I imagine it can be optimized or linked up in someway with creating the
  // OutputFileDir
  SmallString<256> KernelNameFile;
  path::system_temp_directory(true, KernelNameFile);
  path::append(KernelNameFile,
               "KernelNames_" + std::string(Inputs[0].getBaseInput()));
  path::replace_extension(KernelNameFile, "txt",
                                     path::Style::native);
  CmdArgs.push_back(C.getArgs().MakeArgString(KernelNameFile));

  CmdArgs.push_back(InputFileName);

  SmallString<256> tmpDir;
  path::system_temp_directory(true, tmpDir);

  // Work around adding the directory as a temp file so I can retrieve it later
  // in linker phase. This is perhaps a good argument to have everything in the
  // one phase.
  const char *OutputFileDir =
      C.addTempFile(C.getArgs().MakeArgString(path::convert_to_slash(
          Twine{tmpDir + "/" + Inputs[0].getBaseInput() + ".dir"}.str())));

  // generating and pushing a temp directory to our args
  CmdArgs.push_back(OutputFileDir);

  SmallString<128> ExecPath(C.getDriver().Dir);
  path::append(ExecPath, "sycl-xocc");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);

// Inputs that the shell script doesn't use should be ignored
 C.addCommand(llvm::make_unique<Command>(JA, *this,
             Exec, CmdArgs, Inputs));

 // There are multiple outputs from this phase, one per kernel in the file
  return "";
}

///////////////////////////////////////////////////////////////////////////////
////                            XOCC Linker
///////////////////////////////////////////////////////////////////////////////

void SYCL::LinkerXOCC::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  const char *XOCCOutputName = constructXOCCLinkerCommand(C, JA, Output,
                                                          Inputs, Args);

}

const char *SYCL::LinkerXOCC::constructXOCCLinkerCommand(Compilation &C,
                                              const JobAction &JA,
                                              const InputInfo &Output,
                                              const InputInfoList &Inputs,
                                              const llvm::opt::ArgList &Args)
                                              const {
  const auto &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());

  ArgStringList CmdArgs;

  CmdArgs.push_back(
    C.getArgs().MakeArgString(TC.XOCCInstallation.getBinaryPath()));

  CmdArgs.push_back(C.getArgs().MakeArgString("-l"));

  // Long winded way to get the name of the file containing kernel names
  // I imagine it can be optimized or linked up in someway with creating the
  // OutputFileDir
  SmallString<256> KernelNameFile, tmpDir;
  path::system_temp_directory(true, KernelNameFile);
  tmpDir = KernelNameFile;
  path::append(KernelNameFile, "KernelNames_" +
                          std::string(Inputs[0].getBaseInput()));
  path::replace_extension(KernelNameFile, "txt",
                                     path::Style::native);
  CmdArgs.push_back(C.getArgs().MakeArgString(KernelNameFile));
  CmdArgs.push_back(Output.getFilename());

  // Might have to check the filename with multiple file inputs
  for (unsigned i = 0; i < C.getTempFiles().size(); ++i) {
      if (path::extension(C.getTempFiles()[i]) == ".dir")
        CmdArgs.push_back(C.getArgs().MakeArgString(C.getTempFiles()[i]));
  }

  SmallString<128> ExecPath(C.getDriver().Dir);
  path::append(ExecPath, "sycl-xocc");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);

  // Inputs that the shell script doesn't use should be ignored
 C.addCommand(llvm::make_unique<Command>(JA, *this,
              Exec, CmdArgs, Inputs));

  return Output.getFilename();
}

///////////////////////////////////////////////////////////////////////////////
////                            SPIRV Linker
///////////////////////////////////////////////////////////////////////////////

const char *SYCL::Linker::constructLLVMSpirvCommand(Compilation &C,
    const JobAction &JA, const InputInfo &Output, StringRef OutputFilePrefix,
    bool ToBc, const char *InputFileName) const {
  // Construct llvm-spirv command.
  // The output is a bc file or vice versa depending on the -r option usage
  // llvm-spirv -r -o a_kernel.bc a_kernel.spv
  // llvm-spirv -o a_kernel.spv a_kernel.bc
  ArgStringList CmdArgs;
  const char *OutputFileName = nullptr;
  if (ToBc) {
    std::string TmpName =
      C.getDriver().GetTemporaryPath(OutputFilePrefix.str() + "-spirv", "bc");
    OutputFileName = C.addTempFile(C.getArgs().MakeArgString(TmpName));
    CmdArgs.push_back("-r");
    CmdArgs.push_back("-o");
    CmdArgs.push_back(OutputFileName);
  } else {
    CmdArgs.push_back("-spirv-no-deref-attr");
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }
  CmdArgs.push_back(InputFileName);

  SmallString<128> LLVMSpirvPath(C.getDriver().Dir);
  path::append(LLVMSpirvPath, "llvm-spirv");
  const char *LLVMSpirv = C.getArgs().MakeArgString(LLVMSpirvPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, LLVMSpirv, CmdArgs, None));
  return OutputFileName;
}

const char *SYCL::Linker::constructLLVMLinkCommand(
    Compilation &C, const JobAction &JA, StringRef SubArchName,
    StringRef OutputFilePrefix,
    const llvm::opt::ArgStringList &InputFiles) const {
  ArgStringList CmdArgs;
  // Add the input bc's created by compile step.
  for (const auto &II : InputFiles)
    CmdArgs.push_back(II);
  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  SmallString<128> TmpName(C.getDriver().GetTemporaryPath(
                           OutputFilePrefix.str() + "-linked", "bc"));
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpName));
  CmdArgs.push_back(OutputFileName);
  SmallString<128> ExecPath(C.getDriver().Dir);
  path::append(ExecPath, "llvm-link");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, None));
  return OutputFileName;
}

void SYCL::Linker::constructLlcCommand(Compilation &C, const JobAction &JA,
    const InputInfo &Output, const char *InputFileName) const {
  // Construct llc command.
  // The output is an object file
  ArgStringList LlcArgs{"-filetype=obj", "-o",  Output.getFilename(),
                        InputFileName};
  SmallString<128> LlcPath(C.getDriver().Dir);
  path::append(LlcPath, "llc");
  const char *Llc = C.getArgs().MakeArgString(LlcPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, Llc, LlcArgs, None));
}

// For SYCL the inputs of the linker job are SPIR-V binaries and output is
// a single SPIR-V binary.  Input can also be bitcode when specified by
// the user
void SYCL::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");

  std::string SubArchName = getToolChain().getTriple().getArchName();

  // Prefix for temporary file name.
  std::string Prefix = path::stem(SubArchName);

  // We want to use llvm-spirv linker to link spirv binaries before putting
  // them into the fat object.
  // Each command outputs different files.
  ArgStringList SpirvInputs;
  for (const auto &II : Inputs) {
    if (!II.isFilename())
      continue;
    if (Args.hasFlag(options::OPT_fsycl_use_bitcode,
                     options::OPT_fno_sycl_use_bitcode, true))
      SpirvInputs.push_back(II.getFilename());
    else {
      const char *LLVMSpirvOutputFile =
        constructLLVMSpirvCommand(C, JA, Output, Prefix, true,
                                  II.getFilename());
      SpirvInputs.push_back(LLVMSpirvOutputFile);
    }
  }
  const char *LLVMLinkOutputFile =
      constructLLVMLinkCommand(C, JA, SubArchName, Prefix, SpirvInputs);
  constructLLVMSpirvCommand(C, JA, Output, Prefix, false, LLVMLinkOutputFile);
}

///////////////////////////////////////////////////////////////////////////////
////                            SYCL Toolchain
///////////////////////////////////////////////////////////////////////////////

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      XOCCInstallation(D, HostTC.getTriple(), Args),
      isXOCCCompilation(Args.hasArg(options::OPT_fsycl_xocc_device) ? true : false)
{

  if (XOCCInstallation.isValid() && isXOCCCompilation)
    getProgramPaths().push_back(XOCCInstallation.getBinPath());

  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  assert(DeviceOffloadingKind == Action::OFK_SYCL &&
         "Only SYCL offloading kinds are supported");

  CC1Args.push_back("-fsycl-is-device");
}

llvm::opt::DerivedArgList *
SYCLToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    DAL->append(A);
  }

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }
  return DAL;
}

Tool *SYCLToolChain::SelectTool(const JobAction &JA) const {
  switch (JA.getKind()) {
  case Action::LinkJobClass:
    if (!Linker) {
      if (isXOCCCompilation)
        Linker.reset(new tools::SYCL::LinkerXOCC(*this));
      else
        Linker.reset(new tools::SYCL::Linker(*this));
    }
    return Linker.get();
  case Action::AssembleJobClass:
    if (isXOCCCompilation) {
      if (!Assembler)
        Assembler.reset(new tools::SYCL::AssemblerXOCC(*this));
      return Assembler.get();
    }
  default:
    return ToolChain::getTool(JA.getKind());
  }
}

void SYCLToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SYCLToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}
