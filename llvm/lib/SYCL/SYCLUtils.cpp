//===- SYCLUtils.cpp ------------------------------------------------------===//
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

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"

#include "SYCLUtils.h"

namespace llvm {
namespace sycl {

/// Remove a list of attributes from an IR module.
void removeAttributes(Module &M, ArrayRef<Attribute::AttrKind> Kinds) {
  for (auto &F : M.functions())
    for (auto Kind : Kinds) {
      F.removeAttributeAtIndex(AttributeList::FunctionIndex, Kind);
      F.removeAttributeAtIndex(AttributeList::ReturnIndex, Kind);
      for (auto &P : F.args())
        P.removeAttr(Kind);
      for (User *U : F.users())
        if (CallBase *CB = dyn_cast<CallBase>(U)) {
          CB->removeAttributeAtIndex(AttributeList::FunctionIndex, Kind);
          CB->removeAttributeAtIndex(AttributeList::ReturnIndex, Kind);
          for (unsigned int i = 0; i < CB->arg_size(); ++i) {
            CB->removeParamAttr(i, Kind);
          }
        }
    }
}

/// Remove a global metadata from a module.
void removeMetadata(Module &M, StringRef MetadataName) {
  llvm::NamedMDNode *Old = M.getOrInsertNamedMetadata(MetadataName);
  if (Old)
    M.eraseNamedMetadata(Old);
}

/// Replace the function named OldN by the function named NewN then delete the
/// function named OldN.
void replaceFunction(Module &M, StringRef OldN, StringRef NewN) {
  Function *Old = M.getFunction(OldN);
  Function *New = M.getFunction(NewN);
  if (!Old)
    return;
  assert(New);
  assert(Old->getFunctionType() == New->getFunctionType() &&
         "replacement is not possible");
  Old->replaceAllUsesWith(New);
  Old->eraseFromParent();
}

/// Test if a function is a kernel
bool isKernelFunc(const Function *F) {
  return F->getCallingConv() == CallingConv::SPIR_KERNEL ||
         F->hasFnAttribute("fpga.top.func");
}

void annotateKernelFunc(Function *F) {
  F->addFnAttr("fpga.top.func", F->getName());
  F->addFnAttr("fpga.demangled.name", F->getName());
  F->setCallingConv(CallingConv::C);
  F->setLinkage(llvm::GlobalValue::ExternalLinkage);
}

void removeKernelFuncAnnotation(Function *F) {
  F->removeFnAttr("fpga.top.func");
  F->removeFnAttr("fpga.demangled.name");
  F->setCallingConv(CallingConv::C);
  F->setLinkage(llvm::GlobalValue::PrivateLinkage);
}

/// Pipe are represented with 3 sting attributes here are the names:
constexpr const char *xilinx_pipe_type =
    "sycl_xilinx_pipe_type"; // value is "read" or "write"
constexpr const char *xilinx_pipe_id =
    "sycl_xilinx_pipe_id"; // value is the unique ID of the pipe
constexpr const char *xilinx_pipe_depth =
    "sycl_xilinx_pipe_depth"; // value is the depth of the pipe

/// getAttributeAtIndex(0, ...) is the attribute on the return the first argment
/// starts at 1

bool isWritePipe(Argument *Arg) {
  return Arg->getParent()
             ->getAttributeAtIndex(Arg->getArgNo() + 1, sycl::xilinx_pipe_type)
             .getValueAsString() == "write";
}

bool isReadPipe(Argument *Arg) {
  return Arg->getParent()
             ->getAttributeAtIndex(Arg->getArgNo() + 1, sycl::xilinx_pipe_type)
             .getValueAsString() == "read";
}

StringRef getPipeID(Argument *Arg) {
  assert(isPipe(Arg));
  return Arg->getParent()
      ->getAttributeAtIndex(Arg->getArgNo() + 1, sycl::xilinx_pipe_id)
      .getValueAsString();
}

int getPipeDepth(Argument *Arg) {
  assert(isPipe(Arg));
  int val;
  llvm::to_integer(
      Arg->getParent()
          ->getAttributeAtIndex(Arg->getArgNo() + 1, sycl::xilinx_pipe_depth)
          .getValueAsString(),
      val);
  return val;
}

void annotatePipe(Argument *Arg, StringRef Op, StringRef Id, int Depth) {
  Arg->addAttr(
      Attribute::get(Arg->getContext(), sycl::xilinx_pipe_type, Op));
  Arg->addAttr(
      Attribute::get(Arg->getContext(), sycl::xilinx_pipe_id, Id));
  Arg->addAttr(
      Attribute::get(Arg->getContext(), sycl::xilinx_pipe_depth, llvm::formatv("{0}", Depth).str()));
}

void annotateReadPipe(Argument *Arg, StringRef Id, int Depth) {
  annotatePipe(Arg, "read", Id, Depth);
}

void annotateWritePipe(Argument *Arg, StringRef Id, int Depth) {
  annotatePipe(Arg, "write", Id, Depth);
}

void removePipeAnnotation(Argument *Arg) {
  Arg->getParent()->removeParamAttr(Arg->getArgNo(), sycl::xilinx_pipe_id);
  Arg->getParent()->removeParamAttr(Arg->getArgNo(), sycl::xilinx_pipe_type);
  Arg->getParent()->removeParamAttr(Arg->getArgNo(), sycl::xilinx_pipe_depth);
}

/// This function gives llvm::function arguments with no name
/// a default name e.g. arg_0, arg_1..
///
/// This is because if your arguments have no name v++ will commit seppuku
/// when generating XML. Perhaps it's possible to move this to the Clang
/// Frontend by generating the name from the accessor/capture the arguments
/// come from, but I believe it requires a special compiler invocation option
/// to keep arg names from the frontend in the LLVM bitcode.
void giveNameToArguments(Function &F) {
  int Counter = 0;
  for (auto &Arg : F.args()) {
    if (!Arg.hasName())
      Arg.setName("arg_" + Twine{Counter++});
  }
}

} // namespace sycl
} // namespace llvm
