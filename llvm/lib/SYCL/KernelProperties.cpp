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
  static StringRef kindOf(const char *Str) {
    return StringRef(Str, strlen(Str) + 1);
  }
} // namespace

namespace llvm {
KernelProperties::KernelProperties(Function &F) {
  for (Instruction &I : instructions(F)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || CB->getIntrinsicID() != Intrinsic::var_annotation)
      continue;
    auto *Alloca =
        dyn_cast_or_null<AllocaInst>(getUnderlyingObject(CB->getOperand(0)));
    // sycl buffer's property for ddr bank association is lowered
    // as an annotation. As the signature of 
    // llvm.var.annotate takes an i8* as first argument, cast from original 
    // argument type to i8* is done, and the final bitcast is annotated. 
    if (!Alloca)
      continue;
    auto *Str = cast<ConstantDataArray>(
        cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(1)))
            ->getOperand(0));
    if (Str->getRawDataValues() != kindOf("xilinx_ddr_bank"))
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
    if (maxiBundles.find(Bank) == maxiBundles.end()) {
      maxiBundles[Bank] = {formatv("ddrmem{0}", Bank)};
    }
  }
}

Optional<unsigned> KernelProperties::getUserSpecifiedDDRBank(Argument *Arg) {
  for (User *U : Arg->users()) {
    if (auto *Store = dyn_cast<StoreInst>(U))
      if (Store->getValueOperand() == Arg) {
        auto Lookup = userSpecifiedDDRBanks.find(dyn_cast_or_null<AllocaInst>(
            getUnderlyingObject(Store->getPointerOperand())));
        if (Lookup == userSpecifiedDDRBanks.end())
          continue;
        return {Lookup->second};
      }
  }
  return {};
}

Optional<StringRef> KernelProperties::getArgumentMAXIBundle(Argument *Arg) {
  auto DdrId = getUserSpecifiedDDRBank(Arg);
  if (DdrId) {
    return {maxiBundles[DdrId.getValue()].bundleName};
  }
  return {};
}
} // namespace llvm
