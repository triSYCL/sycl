//===--- Chess.cpp - Chess Tool and ToolChain Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Chess.h"
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
////                            Chess Installation Detector
///////////////////////////////////////////////////////////////////////////////

ChessInstallationDetector::ChessInstallationDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args)
    : D(D) {

  /// TODO: Create Chess/Xilinx Cardano Installation Detection Algorithm
  /// look at XOCCInstallationDetector for a general idea.

}

///////////////////////////////////////////////////////////////////////////////
////                            Chess Linker
///////////////////////////////////////////////////////////////////////////////

void SYCL::LinkerChess::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  constructSYCLChessCommand(C, JA, Output, Inputs, Args);
}

void SYCL::LinkerChess::constructSYCLChessCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const llvm::opt::ArgList &Args) const {
  const auto &TC =
    static_cast<const toolchains::ChessToolChain &>(getToolChain());
  /// TODO: Create sycl-chess script invocation with all the required arguments
  /// and then add the invocation command to the compilation. Look at LinkerXOCC
  /// command for an idea.

  //ArgStringList CmdArgs;
  //C.addCommand(llvm::make_unique<Command>(JA, *this,
  //             Exec, CmdArgs, Inputs));
}

///////////////////////////////////////////////////////////////////////////////
////                            Chess Toolchain
///////////////////////////////////////////////////////////////////////////////

ChessToolChain::ChessToolChain(const Driver &D, const llvm::Triple &Triple,
                              const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      ChessInstallation(D, HostTC.getTriple(), Args)
{
  llvm::errs() << "Chess Tool Chain Selected \n";

  if (ChessInstallation.isValid())
    getProgramPaths().push_back(ChessInstallation.getBinPath());

  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void ChessToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  assert(DeviceOffloadingKind == Action::OFK_SYCL &&
         "Only SYCL offloading kinds are supported");

  CC1Args.push_back("-fsycl-is-device");
}

llvm::opt::DerivedArgList *
ChessToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
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

Tool *ChessToolChain::buildLinker() const {
  assert(getTriple().isXilinxAIE());
  return new tools::SYCL::LinkerChess(*this);
}

void ChessToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
ChessToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void ChessToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void ChessToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}
