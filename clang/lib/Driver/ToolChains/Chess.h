//===--- Chess.h - Chess ToolChain Implementation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_CHESS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_CHESS_H

#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"

namespace clang {
namespace driver {

/// Based loosely on CudaInstallationDetector/XOCCInstallationDetector
class ChessInstallationDetector {
private:
  const Driver &D;
  bool IsValid = false;
  std::string BinPath;
  std::string BinaryPath;
  std::string ChessPath;
  std::string LibPath;

public:
  ChessInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                            const llvm::opt::ArgList &Args);

  /// Check whether we detected a valid Chess Install
  bool isValid() const { return IsValid; }

  /// Get the path to the xchesscc binary
  StringRef getBinaryPath() const { return BinaryPath; }

  /// Get the detected path to cardanos's bin directory.
  StringRef getBinPath() const { return BinPath; }

  /// Get the path to Chess compilers root
  StringRef getChessPath() const { return ChessPath; }

  /// Get the detected path to the Chess compilers ME libraries.
  StringRef getLibPath() const { return LibPath; }

};

namespace tools {
namespace SYCL {

class LLVM_LIBRARY_VISIBILITY LinkerChess : public Tool {
public:
  LinkerChess(const ToolChain &TC) : Tool("Chess::LinkerChess",
              "sycl-link-chess", TC) {}

  // technically true, but we don't care about it having integrated C++ for now
  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

  private:
    void constructSYCLChessCommand(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const llvm::opt::ArgList &Args) const;
};

} // end namespace SYCL
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY ChessToolChain : public ToolChain {
public:
  ChessToolChain(const Driver &D, const llvm::Triple &Triple,
                 const ToolChain &HostTC, const llvm::opt::ArgList &Args);

  const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                         llvm::opt::ArgStringList &CC1Args,
                         Action::OffloadKind DeviceOffloadKind) const override;

  bool useIntegratedAs() const override { return true; }

  bool isPICDefault() const override { return false; }
  bool isPIEDefault() const override { return false; }
  bool isPICDefaultForced() const override { return false; }

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CC1Args) const override;

  const ToolChain &HostTC;
  ChessInstallationDetector ChessInstallation;

protected:
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang


#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_CHESS_H
