//===--- SYCL.h - SYCL ToolChain Implementations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H

#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"

namespace clang {
namespace driver {

/// Based loosely on CudaInstallationDetector
class XOCCInstallationDetector {
private:
  const Driver &D;
  bool IsValid = false;
  std::string BinPath;
  std::string BinaryPath;
  std::string SDXPath;
  std::string LibPath;

public:
  XOCCInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                           const llvm::opt::ArgList &Args);

  /// Check whether we detected a valid XOCC Install
  bool isValid() const { return IsValid; }

  /// Get the path to the xocc binary
  StringRef getBinaryPath() const { return BinaryPath; }

  /// Get the detected path to xocc's bin directory.
  StringRef getBinPath() const { return BinPath; }

  /// Get the path to SDX's root, the xocc drivers parent project
  StringRef getSDXPath() const { return SDXPath; }

  /// Get the detected path to xocc's lib directory.
  /// FIXME: This currently assumes lnx64
  StringRef getLibPath() const { return LibPath; }

};

namespace tools {
namespace SYCL {
// Technically this is not an Assemble Stage, it's a Compile Stage.
// However, it fits in after the Clang compiler has compiled the device code to
// IR and outputs an XO (Xilinx Object). So it's hard to say where it fits,
// it seems less intrusive than adding a second Compile stage for the moment.
class LLVM_LIBRARY_VISIBILITY AssemblerXOCC : public Tool {
 public:
   AssemblerXOCC(const ToolChain &TC)
       : Tool("SYCL::AssemblerXOCC", "sycl-assembler-xocc", TC) {}

  // technically true, but we don't care about it having integrated C++ for now
   bool hasIntegratedCPP() const override { return false; }

   void ConstructJob(Compilation &C, const JobAction &JA,
                     const InputInfo &Output, const InputInfoList &Inputs,
                     const llvm::opt::ArgList &TCArgs,
                     const char *LinkingOutput) const override;
  private:
    /// \return opt output file name.
    const char *constructOptCommand(Compilation &C, const JobAction &JA,
                                    const InputInfoList &Inputs,
                                    const llvm::opt::ArgList &Args,
                                    const char *InputFileName) const;

    /// \returns the xocc output file name
    /// Technically it's an XOCC compile stage but it acts sort of like our
    /// assembler, it takes in SPIR .bc and outputs an object file (.xo) of an
    /// individual kernel (it may be possible to generate extra "compilation"
    /// phases for SYCL when compiling for XOCC supported FPGA devices it may be
    /// a lot more work though and requires some further thought)
    const char *constructXOCCCompileCommand(Compilation &C, const JobAction &JA,
                                            const InputInfoList &Inputs,
                                            const llvm::opt::ArgList &Args,
                                            const char *InputFileName) const;

    /// \return llvm-link output file name.
    const char *constructLLVMLinkCommand(Compilation &C, const JobAction &JA,
                                          const InputInfoList &Inputs,
                                          const llvm::opt::ArgList &Args,
                                          const char *InputFileName) const;
};

// Links all of the .xo files (individual kernels) into a final binary blob that
// can be loaded in via the XRT run time
class LLVM_LIBRARY_VISIBILITY LinkerXOCC : public Tool {
public:
  LinkerXOCC(const ToolChain &TC) : Tool("SYCL::LinkerXOCC", "sycl-link-xocc", TC) {}

  // technically true, but we don't care about it having integrated C++ for now
  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

  private:
    const char *constructXOCCLinkerCommand(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const llvm::opt::ArgList &Args)
                                           const;

};

// Runs llvm-spirv to convert spirv to bc, llvm-link, which links multiple LLVM
// bitcode. Converts generated bc back to spirv using llvm-spirv, wraps with
// offloading information. Finally compiles to object using llc
class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("SYCL::Linker", "sycl-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  /// \return llvm-spirv output file name.
  const char *constructLLVMSpirvCommand(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       llvm::StringRef OutputFilePrefix,
                                       bool isBc, const char *InputFile) const;
  /// \return llvm-link output file name.
  const char *constructLLVMLinkCommand(Compilation &C, const JobAction &JA,
                             llvm::StringRef SubArchName,
                             llvm::StringRef OutputFilePrefix,
                             const llvm::opt::ArgStringList &InputFiles) const;
  void constructLlcCommand(Compilation &C, const JobAction &JA,
                           const InputInfo &Output,
                           const char *InputFile) const;
};

} // end namespace SYCL
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY SYCLToolChain : public ToolChain {
public:
  SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
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

  bool useIntegratedAs() const override { return !isXOCCCompilation; }

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
  Tool *SelectTool(const JobAction &JA) const override;

  const ToolChain &HostTC;
  XOCCInstallationDetector XOCCInstallation;

protected:

private:
  // FIXME: having this sort of enunciates the usefulness of having our own
  // triple, especially if we extend to other exotic architectures
  bool isXOCCCompilation;

  mutable std::unique_ptr<Tool> Assembler;
  mutable std::unique_ptr<Tool> Linker;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
