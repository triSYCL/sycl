//===--- Driver.h - Clang GCC Compatible Driver -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_DRIVER_H
#define LLVM_CLANG_DRIVER_DRIVER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Phases.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/StringSaver.h"

#include <list>
#include <map>
#include <string>

namespace llvm {
class Triple;
namespace vfs {
class FileSystem;
}
} // namespace llvm

namespace clang {

namespace driver {

  class Command;
  class Compilation;
  class InputInfo;
  class JobList;
  class JobAction;
  class SanitizerArgs;
  class ToolChain;

/// Describes the kind of LTO mode selected via -f(no-)?lto(=.*)? options.
enum LTOKind {
  LTOK_None,
  LTOK_Full,
  LTOK_Thin,
  LTOK_Unknown
};

/// Driver - Encapsulate logic for constructing compilation processes
/// from a set of gcc-driver-like command line arguments.
class Driver {
  DiagnosticsEngine &Diags;

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS;

  enum DriverMode {
    GCCMode,
    GXXMode,
    CPPMode,
    CLMode,
    FlangMode
  } Mode;

  enum SaveTempsMode {
    SaveTempsNone,
    SaveTempsCwd,
    SaveTempsObj
  } SaveTemps;

  enum BitcodeEmbedMode {
    EmbedNone,
    EmbedMarker,
    EmbedBitcode
  } BitcodeEmbed;

  /// LTO mode selected via -f(no-)?lto(=.*)? options.
  LTOKind LTOMode;

public:
  enum OpenMPRuntimeKind {
    /// An unknown OpenMP runtime. We can't generate effective OpenMP code
    /// without knowing what runtime to target.
    OMPRT_Unknown,

    /// The LLVM OpenMP runtime. When completed and integrated, this will become
    /// the default for Clang.
    OMPRT_OMP,

    /// The GNU OpenMP runtime. Clang doesn't support generating OpenMP code for
    /// this runtime but can swallow the pragmas, and find and link against the
    /// runtime library itself.
    OMPRT_GOMP,

    /// The legacy name for the LLVM OpenMP runtime from when it was the Intel
    /// OpenMP runtime. We support this mode for users with existing
    /// dependencies on this runtime library name.
    OMPRT_IOMP5
  };

  // Diag - Forwarding function for diagnostics.
  DiagnosticBuilder Diag(unsigned DiagID) const {
    return Diags.Report(DiagID);
  }

  // FIXME: Privatize once interface is stable.
public:
  /// The name the driver was invoked as.
  std::string Name;

  /// The path the driver executable was in, as invoked from the
  /// command line.
  std::string Dir;

  /// The original path to the clang executable.
  std::string ClangExecutable;

  /// Target and driver mode components extracted from clang executable name.
  ParsedClangName ClangNameParts;

  /// The path to the installed clang directory, if any.
  std::string InstalledDir;

  /// The path to the compiler resource directory.
  std::string ResourceDir;

  /// System directory for config files.
  std::string SystemConfigDir;

  /// User directory for config files.
  std::string UserConfigDir;

  /// A prefix directory used to emulate a limited subset of GCC's '-Bprefix'
  /// functionality.
  /// FIXME: This type of customization should be removed in favor of the
  /// universal driver when it is ready.
  typedef SmallVector<std::string, 4> prefix_list;
  prefix_list PrefixDirs;

  /// sysroot, if present
  std::string SysRoot;

  /// Dynamic loader prefix, if present
  std::string DyldPrefix;

  /// Driver title to use with help.
  std::string DriverTitle;

  /// Information about the host which can be overridden by the user.
  std::string HostBits, HostMachine, HostSystem, HostRelease;

  /// The file to log CC_PRINT_OPTIONS output to, if enabled.
  const char *CCPrintOptionsFilename;

  /// The file to log CC_PRINT_HEADERS output to, if enabled.
  const char *CCPrintHeadersFilename;

  /// The file to log CC_LOG_DIAGNOSTICS output to, if enabled.
  const char *CCLogDiagnosticsFilename;

  /// A list of inputs and their types for the given arguments.
  typedef SmallVector<std::pair<types::ID, const llvm::opt::Arg *>, 16>
      InputList;

  /// Whether the driver should follow g++ like behavior.
  bool CCCIsCXX() const { return Mode == GXXMode; }

  /// Whether the driver is just the preprocessor.
  bool CCCIsCPP() const { return Mode == CPPMode; }

  /// Whether the driver should follow gcc like behavior.
  bool CCCIsCC() const { return Mode == GCCMode; }

  /// Whether the driver should follow cl.exe like behavior.
  bool IsCLMode() const { return Mode == CLMode; }

  /// Whether the driver should invoke flang for fortran inputs.
  /// Other modes fall back to calling gcc which in turn calls gfortran.
  bool IsFlangMode() const { return Mode == FlangMode; }

  /// Only print tool bindings, don't build any jobs.
  unsigned CCCPrintBindings : 1;

  /// Set CC_PRINT_OPTIONS mode, which is like -v but logs the commands to
  /// CCPrintOptionsFilename or to stderr.
  unsigned CCPrintOptions : 1;

  /// Set CC_PRINT_HEADERS mode, which causes the frontend to log header include
  /// information to CCPrintHeadersFilename or to stderr.
  unsigned CCPrintHeaders : 1;

  /// Set CC_LOG_DIAGNOSTICS mode, which causes the frontend to log diagnostics
  /// to CCLogDiagnosticsFilename or to stderr, in a stable machine readable
  /// format.
  unsigned CCLogDiagnostics : 1;

  /// Whether the driver is generating diagnostics for debugging purposes.
  unsigned CCGenDiagnostics : 1;

  /// Pointer to the ExecuteCC1Tool function, if available.
  /// When the clangDriver lib is used through clang.exe, this provides a
  /// shortcut for executing the -cc1 command-line directly, in the same
  /// process.
  typedef int (*CC1ToolFunc)(SmallVectorImpl<const char *> &ArgV);
  CC1ToolFunc CC1Main = nullptr;

private:
  /// Raw target triple.
  std::string TargetTriple;

  /// Name to use when invoking gcc/g++.
  std::string CCCGenericGCCName;

  /// Name of configuration file if used.
  std::string ConfigFile;

  /// Allocator for string saver.
  llvm::BumpPtrAllocator Alloc;

  /// Object that stores strings read from configuration file.
  llvm::StringSaver Saver;

  /// Arguments originated from configuration file.
  std::unique_ptr<llvm::opt::InputArgList> CfgOptions;

  /// Arguments originated from command line.
  std::unique_ptr<llvm::opt::InputArgList> CLOptions;

  /// Whether to check that input files exist when constructing compilation
  /// jobs.
  unsigned CheckInputsExist : 1;

public:
  /// Force clang to emit reproducer for driver invocation. This is enabled
  /// indirectly by setting FORCE_CLANG_DIAGNOSTICS_CRASH environment variable
  /// or when using the -gen-reproducer driver flag.
  unsigned GenReproducer : 1;

private:
  /// Certain options suppress the 'no input files' warning.
  unsigned SuppressMissingInputWarning : 1;

  /// Cache of all the ToolChains in use by the driver.
  ///
  /// This maps from the string representation of a triple to a ToolChain
  /// created targeting that triple. The driver owns all the ToolChain objects
  /// stored in it, and will clean them up when torn down.
  mutable llvm::StringMap<std::unique_ptr<ToolChain>> ToolChains;

private:
  /// TranslateInputArgs - Create a new derived argument list from the input
  /// arguments, after applying the standard argument translations.
  llvm::opt::DerivedArgList *
  TranslateInputArgs(const llvm::opt::InputArgList &Args) const;

  // getFinalPhase - Determine which compilation mode we are in and record
  // which option we used to determine the final phase.
  // TODO: Much of what getFinalPhase returns are not actually true compiler
  //       modes. Fold this functionality into Types::getCompilationPhases and
  //       handleArguments.
  phases::ID getFinalPhase(const llvm::opt::DerivedArgList &DAL,
                           llvm::opt::Arg **FinalPhaseArg = nullptr) const;

  // handleArguments - All code related to claiming and printing diagnostics
  // related to arguments to the driver are done here.
  void handleArguments(Compilation &C, llvm::opt::DerivedArgList &Args,
                       const InputList &Inputs, ActionList &Actions) const;

  // Before executing jobs, sets up response files for commands that need them.
  void setUpResponseFiles(Compilation &C, Command &Cmd);

  void generatePrefixedToolNames(StringRef Tool, const ToolChain &TC,
                                 SmallVectorImpl<std::string> &Names) const;

  /// Find the appropriate .crash diagonostic file for the child crash
  /// under this driver and copy it out to a temporary destination with the
  /// other reproducer related files (.sh, .cache, etc). If not found, suggest a
  /// directory for the user to look at.
  ///
  /// \param ReproCrashFilename The file path to copy the .crash to.
  /// \param CrashDiagDir       The suggested directory for the user to look at
  ///                           in case the search or copy fails.
  ///
  /// \returns If the .crash is found and successfully copied return true,
  /// otherwise false and return the suggested directory in \p CrashDiagDir.
  bool getCrashDiagnosticFile(StringRef ReproCrashFilename,
                              SmallString<128> &CrashDiagDir);

public:

  /// Takes the path to a binary that's either in bin/ or lib/ and returns
  /// the path to clang's resource directory.
  static std::string GetResourcesPath(StringRef BinaryPath,
                                      StringRef CustomResourceDir = "");

  Driver(StringRef ClangExecutable, StringRef TargetTriple,
         DiagnosticsEngine &Diags,
         IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS = nullptr);

  /// @name Accessors
  /// @{

  /// Name to use when invoking gcc/g++.
  const std::string &getCCCGenericGCCName() const { return CCCGenericGCCName; }

  const std::string &getConfigFile() const { return ConfigFile; }

  const llvm::opt::OptTable &getOpts() const { return getDriverOptTable(); }

  const DiagnosticsEngine &getDiags() const { return Diags; }

  llvm::vfs::FileSystem &getVFS() const { return *VFS; }

  bool getCheckInputsExist() const { return CheckInputsExist; }

  void setCheckInputsExist(bool Value) { CheckInputsExist = Value; }

  void setTargetAndMode(const ParsedClangName &TM) { ClangNameParts = TM; }

  const std::string &getTitle() { return DriverTitle; }
  void setTitle(std::string Value) { DriverTitle = std::move(Value); }

  std::string getTargetTriple() const { return TargetTriple; }

  /// Get the path to the main clang executable.
  const char *getClangProgramPath() const {
    return ClangExecutable.c_str();
  }

  /// Get the path to where the clang executable was installed.
  const char *getInstalledDir() const {
    if (!InstalledDir.empty())
      return InstalledDir.c_str();
    return Dir.c_str();
  }
  void setInstalledDir(StringRef Value) { InstalledDir = std::string(Value); }

  bool isSaveTempsEnabled() const { return SaveTemps != SaveTempsNone; }
  bool isSaveTempsObj() const { return SaveTemps == SaveTempsObj; }

  bool embedBitcodeEnabled() const { return BitcodeEmbed != EmbedNone; }
  bool embedBitcodeInObject() const { return (BitcodeEmbed == EmbedBitcode); }
  bool embedBitcodeMarkerOnly() const { return (BitcodeEmbed == EmbedMarker); }

  /// Compute the desired OpenMP runtime from the flags provided.
  OpenMPRuntimeKind getOpenMPRuntime(const llvm::opt::ArgList &Args) const;

  /// @}
  /// @name Primary Functionality
  /// @{

  /// CreateOffloadingDeviceToolChains - create all the toolchains required to
  /// support offloading devices given the programming models specified in the
  /// current compilation. Also, update the host tool chain kind accordingly.
  void CreateOffloadingDeviceToolChains(Compilation &C, InputList &Inputs);

  /// BuildCompilation - Construct a compilation object for a command
  /// line argument vector.
  ///
  /// \return A compilation, or 0 if none was built for the given
  /// argument vector. A null return value does not necessarily
  /// indicate an error condition, the diagnostics should be queried
  /// to determine if an error occurred.
  Compilation *BuildCompilation(ArrayRef<const char *> Args);

  /// @name Driver Steps
  /// @{

  /// ParseDriverMode - Look for and handle the driver mode option in Args.
  void ParseDriverMode(StringRef ProgramName, ArrayRef<const char *> Args);

  /// ParseArgStrings - Parse the given list of strings into an
  /// ArgList.
  llvm::opt::InputArgList ParseArgStrings(ArrayRef<const char *> Args,
                                          bool IsClCompatMode,
                                          bool &ContainsError);

  /// BuildInputs - Construct the list of inputs and their types from
  /// the given arguments.
  ///
  /// \param TC - The default host tool chain.
  /// \param Args - The input arguments.
  /// \param Inputs - The list to store the resulting compilation
  /// inputs onto.
  void BuildInputs(const ToolChain &TC, llvm::opt::DerivedArgList &Args,
                   InputList &Inputs) const;

  /// BuildActions - Construct the list of actions to perform for the
  /// given arguments, which are only done for a single architecture.
  ///
  /// \param C - The compilation that is being built.
  /// \param Args - The input arguments.
  /// \param Actions - The list to store the resulting actions onto.
  void BuildActions(Compilation &C, llvm::opt::DerivedArgList &Args,
                    const InputList &Inputs, ActionList &Actions) const;

  /// BuildUniversalActions - Construct the list of actions to perform
  /// for the given arguments, which may require a universal build.
  ///
  /// \param C - The compilation that is being built.
  /// \param TC - The default host tool chain.
  void BuildUniversalActions(Compilation &C, const ToolChain &TC,
                             const InputList &BAInputs) const;

  /// Check that the file referenced by Value exists. If it doesn't,
  /// issue a diagnostic and return false.
  /// If TypoCorrect is true and the file does not exist, see if it looks
  /// like a likely typo for a flag and if so print a "did you mean" blurb.
  bool DiagnoseInputExistence(const llvm::opt::DerivedArgList &Args,
                              StringRef Value, types::ID Ty,
                              bool TypoCorrect) const;

  /// BuildJobs - Bind actions to concrete tools and translate
  /// arguments to form the list of jobs to run.
  ///
  /// \param C - The compilation that is being built.
  void BuildJobs(Compilation &C) const;

  /// ExecuteCompilation - Execute the compilation according to the command line
  /// arguments and return an appropriate exit code.
  ///
  /// This routine handles additional processing that must be done in addition
  /// to just running the subprocesses, for example reporting errors, setting
  /// up response files, removing temporary files, etc.
  int ExecuteCompilation(Compilation &C,
     SmallVectorImpl< std::pair<int, const Command *> > &FailingCommands);

  /// Contains the files in the compilation diagnostic report generated by
  /// generateCompilationDiagnostics.
  struct CompilationDiagnosticReport {
    llvm::SmallVector<std::string, 4> TemporaryFiles;
  };

  /// generateCompilationDiagnostics - Generate diagnostics information
  /// including preprocessed source file(s).
  ///
  void generateCompilationDiagnostics(
      Compilation &C, const Command &FailingCommand,
      StringRef AdditionalInformation = "",
      CompilationDiagnosticReport *GeneratedReport = nullptr);

  /// @}
  /// @name Helper Methods
  /// @{

  /// MakeSYCLDeviceTriple - Returns the SYCL device triple for the
  /// specified subarch
  llvm::Triple MakeSYCLDeviceTriple(StringRef TargetArch = "spir64") const;

  /// PrintActions - Print the list of actions.
  void PrintActions(const Compilation &C) const;

  /// PrintHelp - Print the help text.
  ///
  /// \param ShowHidden - Show hidden options.
  void PrintHelp(bool ShowHidden) const;

  /// PrintSYCLToolHelp - Print help text from offline compiler tools.
  void PrintSYCLToolHelp(const Compilation &C) const;

  /// PrintVersion - Print the driver version.
  void PrintVersion(const Compilation &C, raw_ostream &OS) const;

  /// GetFilePath - Lookup \p Name in the list of file search paths.
  ///
  /// \param TC - The tool chain for additional information on
  /// directories to search.
  //
  // FIXME: This should be in CompilationInfo.
  std::string GetFilePath(StringRef Name, const ToolChain &TC) const;

  /// GetProgramPath - Lookup \p Name in the list of program search paths.
  ///
  /// \param TC - The provided tool chain for additional information on
  /// directories to search.
  //
  // FIXME: This should be in CompilationInfo.
  std::string GetProgramPath(StringRef Name, const ToolChain &TC) const;

  /// HandleAutocompletions - Handle --autocomplete by searching and printing
  /// possible flags, descriptions, and its arguments.
  void HandleAutocompletions(StringRef PassedFlags) const;

  /// HandleImmediateArgs - Handle any arguments which should be
  /// treated before building actions or binding tools.
  ///
  /// \return Whether any compilation should be built for this
  /// invocation.
  bool HandleImmediateArgs(const Compilation &C);

  /// ConstructAction - Construct the appropriate action to do for
  /// \p Phase on the \p Input, taking in to account arguments
  /// like -fsyntax-only or --analyze.
  Action *ConstructPhaseAction(
      Compilation &C, const llvm::opt::ArgList &Args, phases::ID Phase,
      Action *Input,
      Action::OffloadKind TargetDeviceOffloadKind = Action::OFK_None) const;

  /// BuildJobsForAction - Construct the jobs to perform for the action \p A and
  /// return an InputInfo for the result of running \p A.  Will only construct
  /// jobs for a given (Action, ToolChain, BoundArch, DeviceKind) tuple once.
  InputInfo
  BuildJobsForAction(Compilation &C, const Action *A, const ToolChain *TC,
                     StringRef BoundArch, bool AtTopLevel, bool MultipleArchs,
                     const char *LinkingOutput,
                     std::map<std::pair<const Action *, std::string>, InputInfo>
                         &CachedResults,
                     Action::OffloadKind TargetDeviceOffloadKind) const;

  /// Returns the default name for linked images (e.g., "a.out").
  const char *getDefaultImageName() const;

  /// GetNamedOutputPath - Return the name to use for the output of
  /// the action \p JA. The result is appended to the compilation's
  /// list of temporary or result files, as appropriate.
  ///
  /// \param C - The compilation.
  /// \param JA - The action of interest.
  /// \param BaseInput - The original input file that this action was
  /// triggered by.
  /// \param BoundArch - The bound architecture.
  /// \param AtTopLevel - Whether this is a "top-level" action.
  /// \param MultipleArchs - Whether multiple -arch options were supplied.
  /// \param NormalizedTriple - The normalized triple of the relevant target.
  const char *GetNamedOutputPath(Compilation &C, const JobAction &JA,
                                 const char *BaseInput, StringRef BoundArch,
                                 bool AtTopLevel, bool MultipleArchs,
                                 StringRef NormalizedTriple) const;

  /// GetTemporaryPath - Return the pathname of a temporary file to use
  /// as part of compilation; the file will have the given prefix and suffix.
  ///
  /// GCC goes to extra lengths here to be a bit more robust.
  std::string GetTemporaryPath(StringRef Prefix, StringRef Suffix) const;

  /// GetUniquePath = Return the pathname of a unique file to use
  /// as part of compilation. The file will have the given base name (BaseName)
  /// and extension (Ext).
  std::string GetUniquePath(StringRef BaseName, StringRef Ext) const;

  /// GetTemporaryDirectory - Return the pathname of a temporary directory to
  /// use as part of compilation; the directory will have the given prefix.
  std::string GetTemporaryDirectory(StringRef Prefix) const;

  /// Return the pathname of the pch file in clang-cl mode.
  std::string GetClPchPath(Compilation &C, StringRef BaseName) const;

  /// ShouldUseClangCompiler - Should the clang compiler be used to
  /// handle this action.
  bool ShouldUseClangCompiler(const JobAction &JA) const;

  /// ShouldUseFlangCompiler - Should the flang compiler be used to
  /// handle this action.
  bool ShouldUseFlangCompiler(const JobAction &JA) const;

  /// ShouldEmitStaticLibrary - Should the linker emit a static library.
  bool ShouldEmitStaticLibrary(const llvm::opt::ArgList &Args) const;

  /// Returns true if we are performing any kind of LTO.
  bool isUsingLTO() const { return LTOMode != LTOK_None; }

  /// Get the specific kind of LTO being performed.
  LTOKind getLTOMode() const { return LTOMode; }

private:

  /// Tries to load options from configuration file.
  ///
  /// \returns true if error occurred.
  bool loadConfigFile();

  /// Read options from the specified file.
  ///
  /// \param [in] FileName File to read.
  /// \returns true, if error occurred while reading.
  bool readConfigFile(StringRef FileName);

  /// Set the driver mode (cl, gcc, etc) from an option string of the form
  /// --driver-mode=<mode>.
  void setDriverModeFromOption(StringRef Opt);

  /// Parse the \p Args list for LTO options and record the type of LTO
  /// compilation based on which -f(no-)?lto(=.*)? option occurs last.
  void setLTOMode(const llvm::opt::ArgList &Args);

  /// Retrieves a ToolChain for a particular \p Target triple.
  ///
  /// Will cache ToolChains for the life of the driver object, and create them
  /// on-demand.
  const ToolChain &getToolChain(const llvm::opt::ArgList &Args,
                                const llvm::Triple &Target) const;

  /// @}

  /// Retrieves a ToolChain for a particular device \p Target triple
  ///
  /// \param[in] HostTC is the host ToolChain paired with the device
  ///
  /// \param[in] Action (e.g. OFK_Cuda/OFK_OpenMP/OFK_SYCL) is an Offloading
  /// action that is optionally passed to a ToolChain (used by CUDA, to specify
  /// if it's used in conjunction with OpenMP)
  ///
  /// Will cache ToolChains for the life of the driver object, and create them
  /// on-demand.
  const ToolChain &getOffloadingDeviceToolChain(const llvm::opt::ArgList &Args,
                                                const llvm::Triple &Target,
                                                const ToolChain &HostTC,
                                                const Action::OffloadKind
                                                &TargetDeviceOffloadKind) const;

  /// Get bitmasks for which option flags to include and exclude based on
  /// the driver mode.
  std::pair<unsigned, unsigned> getIncludeExcludeOptionFlagMasks(bool IsClCompatMode) const;

  /// Helper used in BuildJobsForAction.  Doesn't use the cache when building
  /// jobs specifically for the given action, but will use the cache when
  /// building jobs for the Action's inputs.
  InputInfo BuildJobsForActionNoCache(
      Compilation &C, const Action *A, const ToolChain *TC, StringRef BoundArch,
      bool AtTopLevel, bool MultipleArchs, const char *LinkingOutput,
      std::map<std::pair<const Action *, std::string>, InputInfo>
          &CachedResults,
      Action::OffloadKind TargetDeviceOffloadKind) const;

  /// Static offload library seen.
  bool OffloadStaticLibSeen = false;

  void setOffloadStaticLibSeen() { OffloadStaticLibSeen = true; }

  /// Returns true if an offload static library is found.
  bool checkForOffloadStaticLib(Compilation &C,
                                llvm::opt::DerivedArgList &Args) const;

  /// Track filename used for the FPGA dependency info.
  mutable llvm::StringMap<const std::string> FPGATempDepFiles;

public:
  /// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and
  /// return the grouped values as integers. Numbers which are not
  /// provided are set to 0.
  ///
  /// \return True if the entire string was parsed (9.2), or all
  /// groups were parsed (10.3.5extrastuff). HadExtra is true if all
  /// groups were parsed but extra characters remain at the end.
  static bool GetReleaseVersion(StringRef Str, unsigned &Major, unsigned &Minor,
                                unsigned &Micro, bool &HadExtra);

  /// Parse digits from a string \p Str and fulfill \p Digits with
  /// the parsed numbers. This method assumes that the max number of
  /// digits to look for is equal to Digits.size().
  ///
  /// \return True if the entire string was parsed and there are
  /// no extra characters remaining at the end.
  static bool GetReleaseVersion(StringRef Str,
                                MutableArrayRef<unsigned> Digits);
  /// Compute the default -fmodule-cache-path.
  /// \return True if the system provides a default cache directory.
  static bool getDefaultModuleCachePath(SmallVectorImpl<char> &Result);

  bool getOffloadStaticLibSeen() const { return OffloadStaticLibSeen; };

  /// addFPGATempDepFile - Add a file to be added to the bundling step of
  /// an FPGA object.
  void addFPGATempDepFile(const std::string &DepName,
                          const std::string &FileName) const {
    FPGATempDepFiles.insert({FileName, DepName});
  }
  /// getFPGATempDepFile - Get a file to be added to the bundling step of
  /// an FPGA object.
  const std::string getFPGATempDepFile(const std::string &FileName) const {
    return FPGATempDepFiles[FileName];
  }
};

/// \return True if the last defined optimization level is -Ofast.
/// And False otherwise.
bool isOptimizationLevelFast(const llvm::opt::ArgList &Args);

/// \return True if the filename has a valid object file extension.
bool isObjectFile(std::string FileName);

/// \return True if the filename has a static archive/lib extension.
bool isStaticArchiveFile(const StringRef &FileName);

/// \return True if the argument combination will end up generating remarks.
bool willEmitRemarks(const llvm::opt::ArgList &Args);

} // end namespace driver
} // end namespace clang

#endif
