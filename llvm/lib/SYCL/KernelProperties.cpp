//===- KernelProperties.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utility functions that read sycl kernel properties
// ===---------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/SYCL/KernelProperties.h"

using namespace llvm;
namespace {
  static StringRef KindOf(const char *Str) {
    return StringRef(Str, strlen(Str) + 1);
  }
}

namespace llvm {
KernelProperties::KernelProperties(Function &F) {
  for (Instruction &I : instructions(F)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || CB->getIntrinsicID() != Intrinsic::var_annotation)
      continue;
    auto *Alloca =
        dyn_cast_or_null<AllocaInst>(getUnderlyingObject(CB->getOperand(0)));
    auto *Str = cast<ConstantDataArray>(
        cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(1)))
            ->getOperand(0));
    if (!Alloca)
      continue;
    if (Str->getRawDataValues() != KindOf("xilinx_ddr_bank"))
      continue;
    Constant *Args =
        (cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(4)))
             ->getInitializer());
    unsigned Bank;
    if (auto *ZeroData = dyn_cast<ConstantAggregateZero>(Args))
      Bank = 0;
    else
      Bank = cast<ConstantInt>(Args->getOperand(0))->getZExtValue();

    userSpecifiedDDRBanks[Alloca] = Bank;
  }
}

unsigned KernelProperties::getUserSpecifiedDDRBank(Argument *Arg) {
  for (User *U : Arg->users()) {
    if (auto *Store = dyn_cast<StoreInst>(U))
      if (Store->getValueOperand() == Arg) {
        auto Lookup = userSpecifiedDDRBanks.find(dyn_cast_or_null<AllocaInst>(
            getUnderlyingObject(Store->getPointerOperand())));
        if (Lookup == userSpecifiedDDRBanks.end())
          continue;
        return Lookup->second;
      }
  }
  return 0;
}
}