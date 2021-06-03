//===- DownGradeUtils.h                                     ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools to downgrad IR
//
// ===---------------------------------------------------------------------===//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"

namespace llvm {

inline void removeAttributes(Module &M, ArrayRef<Attribute::AttrKind> Kinds) {
  for (auto &F : M.functions())
    for (auto Kind : Kinds) {
      F.removeAttribute(AttributeList::FunctionIndex, Kind);
      F.removeAttribute(AttributeList::ReturnIndex, Kind);
      for (auto &P : F.args())
        P.removeAttr(Kind);
      for (User *U : F.users())
        if (CallBase *CB = dyn_cast<CallBase>(U)) {
          CB->removeAttribute(AttributeList::FunctionIndex, Kind);
          CB->removeAttribute(AttributeList::ReturnIndex, Kind);
          for (unsigned int i = 0; i < CB->getNumArgOperands(); ++i) {
            CB->removeParamAttr(i, Kind);
          }
        }
    }
}

}
