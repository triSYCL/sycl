//===--- VXX.h - V++ ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_VXX_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_VXX_H

#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"
#include "llvm/ADT/Triple.h"

namespace clang {
namespace driver {

/// Based loosely on CudaInstallationDetector
class VXXInstallationDetector {
private:
  bool IsValid = false;
  std::string BinPath;
  std::string BinaryPath;
  std::string VitisPath;
  std::string LibPath;

public:
  VXXInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                           const llvm::opt::ArgList &Args);

  /// Check whether we detected a valid v++ installation
  bool isValid() const { return IsValid; }

  /// Get the path to the v++ binary
  StringRef getBinaryPath() const { return BinaryPath; }

  /// Get the detected path to v++'s bin directory.
  StringRef getBinPath() const { return BinPath; }

  /// Get the path to Vitis's root, the v++ drivers parent project
  StringRef getVitisPath() const { return VitisPath; }

  /// Get the detected path to v++'s lib directory.
  /// FIXME: This currently assumes lnx64
  StringRef getLibPath() const { return LibPath; }

};

// \todo come up with a better name like,  SYCLAssemblerVXX/Linker for the
// tools? Or should the tool just be SYCLVXXToolchain?
namespace tools {
namespace SYCL {


// Technically this is not just a Linker Stage, it's a Compile and Linker Stage.
// However, it fits in after the Clang compiler has compiled the device code to
// BC and allows us to compile to an xcl binary to be offloaded. It's less
// complex and intrusive than optionally altering the SYCL offloader phases
// based on target and is similar to what the existing SYCL ToolChain does.
//
// Compiles all the kernels into .xo files and then links all of the .xo files
// (individual kernels) into a final binary blob that can be offloaded and
// wrapped into the final binary. Which XRT can then load and execute like a
// normal pre-compiled OpenCL binary.
class LLVM_LIBRARY_VISIBILITY LinkerVXX : public Tool {
public:
  LinkerVXX(const ToolChain &TC) : Tool("VXX::LinkerVXX", "sycl-link-vxx", TC) {}

  // technically true, but we don't care about it having integrated C++ for now
  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

  private:
    void constructSYCLVXXCommand(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const llvm::opt::ArgList &Args) const;
};

class LLVM_LIBRARY_VISIBILITY SYCLPostLinkVXX : public Tool {
public:
  SYCLPostLinkVXX(const ToolChain &TC)
      : Tool("VXX::SYCLPostLinkVXX", "sycl-post-link-vxx", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  void constructSYCLVXXPLCommand(Compilation &C, const JobAction &JA,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const llvm::opt::ArgList &Args) const;
};

} // end namespace SYCL
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY VXXToolChain : public ToolChain {
public:
  VXXToolChain(const Driver &D, const llvm::Triple &Triple,
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

  // \todo change when remove assembler
  bool useIntegratedAs() const override { return true; }

  bool isPICDefault() const override { return false; }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override { return false; }
  bool isPICDefaultForced() const override { return false; }

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CC1Args) const override;
  // Tool *SelectTool(const JobAction &JA) const override;

  const ToolChain &HostTC;
  VXXInstallationDetector VXXInstallation;

protected:
  Tool *buildLinker() const override;
  virtual Tool *getTool(Action::ActionClass AC) const override;

private:
  mutable std::unique_ptr<Tool> VXXSYCLPostLink;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang


#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_VXX_H
