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

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "SYCLUtils.h"

namespace llvm {

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

} // namespace llvm
