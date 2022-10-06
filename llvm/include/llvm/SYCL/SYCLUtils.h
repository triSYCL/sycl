//===- SYCLUtils.h --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared utilities between the various SYCL passes.
//
// ===---------------------------------------------------------------------===//

#ifndef LLVM_SYCL_SYCLUTILS_H
#define LLVM_SYCL_SYCLUTILS_H

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace sycl {

/// Remove a list of attributes from an IR module.
void removeAttributes(Module &M, ArrayRef<Attribute::AttrKind> Kinds);

/// Remove a global metadata from a module.
void removeMetadata(Module &M, StringRef MetadataName);

/// Replace the function named OldN by the function named NewN then delete the
/// function named OldN.
void replaceFunction(Module &M, StringRef OldN, StringRef NewN);

/// Test if a function is a kernel
bool isKernelFunc(const Function *F);

/// Test if the argument is a buffer in the OpenCL sense
bool isArgBuffer(Argument *Arg);

/// Add annotation such that F is considered a Kernel by our passes and
/// Vitis's HLS
void annotateKernelFunc(Function *F);

/// Remove annotation that make F a kernel
void removeKernelFuncAnnotation(Function *F);

/// Return true iff Arg is a pipe for writing
bool isWritePipe(Argument *Arg);

/// Return true iff Arg is a pipe for reading
bool isReadPipe(Argument *Arg);

/// Return true iff Arg is a pipe
inline bool isPipe(Argument *Arg) {
  return isWritePipe(Arg) || isReadPipe(Arg);
}

/// Return the Identifier of a pipe
StringRef getPipeID(Argument *Arg);

/// Return the Depth of a pipe
int getPipeDepth(Argument *Arg);

/// Add annotation such that Arg is considered a read pipe
void annotateReadPipe(Argument *Arg, StringRef Id, int Depth);

/// Add annotation such that Arg is considered a write pipe
void annotateWritePipe(Argument *Arg, StringRef Id, int Depth);

/// Remove annotations that make Arg a pipe
void removePipeAnnotation(Argument *Arg);

/// Rename arguments to comply with Vitis's HLS
void giveNameToArguments(Function &F);

enum struct MemoryType { unspecified, ddr, hbm };

struct MemBankSpec {
  MemoryType MemType;
  unsigned BankID;
  operator bool() const { return MemType != MemoryType::unspecified; }
};

void annotateMemoryBank(Argument *Arg, MemBankSpec Val);
void removeMemoryBankAnnotation(Argument *Arg);

MemBankSpec getMemoryBank(Argument *Arg);

} // namespace sycl
} // namespace llvm

#endif
