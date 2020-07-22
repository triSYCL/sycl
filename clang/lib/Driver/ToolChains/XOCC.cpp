//===--- XOCC.cpp - XOCC Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "XOCC.h"
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
  auto search_and_set_up_program = [&] (const char * programName) {
    llvm::ErrorOr<std::string> program = findProgramByName(programName);
    if (program) {
      SmallString<256> programsAbsolutePath;
      fs::real_path(*program, programsAbsolutePath);

      BinaryPath = programsAbsolutePath.str();

      StringRef programDir = path::parent_path(programsAbsolutePath);

      if (path::filename(programDir) == "bin")
        BinPath = programDir;

      // TODO: Check if this assumption is correct in all installations and give
      // environment variable specifier option or an argument to the Driver
      SDXPath = path::parent_path(programDir);
      LibPath = SDXPath + "/lnx64/lib";

      // TODO: slightly stricter IsValid test... check all strings aren't empty
      IsValid = true;
    }
    return program;
  };

  if (!search_and_set_up_program("v++"))
    search_and_set_up_program("xocc");
}

///////////////////////////////////////////////////////////////////////////////
////                            XOCC Linker
///////////////////////////////////////////////////////////////////////////////

void SYCL::LinkerXOCC::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  constructSYCLXOCCCommand(C, JA, Output, Inputs, Args);
}

// \todo: Extend to support the possibility of more than one file being passed
// to the linker stage
// \todo: Add additional modifications that were added to the SYCL ToolChain
// recently if feasible
void SYCL::LinkerXOCC::constructSYCLXOCCCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const llvm::opt::ArgList &Args) const {
  const auto &TC =
    static_cast<const toolchains::XOCCToolChain &>(getToolChain());

  ArgStringList CmdArgs;

  // Script Arg $1, directory of xocc binary (SDx's bin)
  CmdArgs.push_back(Args.MakeArgString(TC.XOCCInstallation.getBinPath()));

  // Script Arg $2, directory of the Clang driver, where the sycl-xocc script
  // opt binary and llvm-linker binary should be contained among other things
  CmdArgs.push_back(Args.MakeArgString(C.getDriver().Dir));

  // Script Arg $3, the original source file name minus the file extension
  // (.h/.cpp etc)
  SmallString<256> SrcName =
    llvm::sys::path::filename(Inputs[0].getBaseInput());
  llvm::sys::path::replace_extension(SrcName, "");
  CmdArgs.push_back(Args.MakeArgString(SrcName));

  // Script Arg $4, input file name, distinct from Arg $3 as this is the .o
  // (it's actually a .bc file in disguise at the moment) input file with a
  // mangled temporary name
  // \todo support more than one input, there may be multiple in some cases as
  // this is a "linker" stage, can refer to SYCL.cpp for an example
  CmdArgs.push_back(Args.MakeArgString(Inputs[0].getFilename()));

  // Script Arg $5, temporary directory path, used to dump a lot of intermediate
  // files that no one needs to know about unless they're debugging
  SmallString<256> TmpDir;
  llvm::sys::path::system_temp_directory(true, TmpDir);
  CmdArgs.push_back(Args.MakeArgString(TmpDir));

  // Script Arg $6, the name of the final output .xcl binary file after
  // compilation and linking is complete
  CmdArgs.push_back(Output.getFilename());

  // Path to sycl-xocc script
  SmallString<128> ExecPath(C.getDriver().Dir);
  path::append(ExecPath, "sycl-xocc");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);

  // Generate our command to sycl-xocc using the arguments we've made
  // Note: Inputs that the shell script doesn't use should be ignored
  C.addCommand(std::make_unique<Command>(JA, *this,
               Exec, CmdArgs, Inputs));
}

///////////////////////////////////////////////////////////////////////////////
////                            XOCC Toolchain
///////////////////////////////////////////////////////////////////////////////

XOCCToolChain::XOCCToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      XOCCInstallation(D, HostTC.getTriple(), Args)
{

  if (XOCCInstallation.isValid())
    getProgramPaths().push_back(XOCCInstallation.getBinPath());

  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void XOCCToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  assert(DeviceOffloadingKind == Action::OFK_SYCL &&
         "Only SYCL offloading kinds are supported");

  CC1Args.push_back("-fsycl-is-device");
}

llvm::opt::DerivedArgList *
XOCCToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
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

Tool *XOCCToolChain::buildLinker() const {
  assert(getTriple().isXilinxFPGA());
  return new tools::SYCL::LinkerXOCC(*this);
}

void XOCCToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
XOCCToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void XOCCToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void XOCCToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}
