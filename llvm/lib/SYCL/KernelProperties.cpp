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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/SYCL/KernelProperties.h"
#include "llvm/Support/Casting.h"

using namespace llvm;
namespace {
enum SPIRAddressSpace {
  SPIRAS_Private,  // Address space: 0
  SPIRAS_Global,   // Address space: 1
  SPIRAS_Constant, // Address space: 2
  SPIRAS_Local,    // Address space: 3
  SPIRAS_Generic,  // Address space: 4
};

static StringRef kindOf(const char *Str) {
  return StringRef(Str, strlen(Str) + 1);
  }

  void collectUserSpecifiedDDRBanks(
      Function &F,
      SmallDenseMap<llvm::AllocaInst *, unsigned, 16> &UserSpecifiedDDRBanks) {
    for (Instruction &I : instructions(F)) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || CB->getIntrinsicID() != Intrinsic::var_annotation)
        continue;
      auto *Alloca =
          dyn_cast_or_null<AllocaInst>(getUnderlyingObject(CB->getOperand(0)));
      // SYCL buffer's property for DDR bank association is lowered
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

      UserSpecifiedDDRBanks[Alloca] = Bank;
    }
  }

  Optional<unsigned> getUserSpecifiedDDRBank(Argument *Arg, SmallDenseMap<llvm::AllocaInst *, unsigned, 16> &UserSpecifiedDDRBanks) {
  for (User *U : Arg->users()) {
    if (auto *Store = dyn_cast<StoreInst>(U))
      if (Store->getValueOperand() == Arg) {
        auto Lookup = UserSpecifiedDDRBanks.find(dyn_cast_or_null<AllocaInst>(
            getUnderlyingObject(Store->getPointerOperand())));
        if (Lookup == UserSpecifiedDDRBanks.end())
          continue;
        return {Lookup->second};
      }
  }
  return {};
}
} // namespace

namespace llvm {
bool KernelProperties::isArgBuffer(Argument* Arg, bool SyclHLSFlow) {
  if (Arg->getType()->isPointerTy() &&
      (SyclHLSFlow ||
       Arg->getType()->getPointerAddressSpace() == SPIRAS_Global ||
       Arg->getType()->getPointerAddressSpace() == SPIRAS_Constant)) {
    return !Arg->hasByValAttr();
  }
  return false;
}

KernelProperties::KernelProperties(Function &F, bool SyclHlsFlow) {

  SmallDenseMap<llvm::AllocaInst*, unsigned, 16> UserSpecifiedDDRBanks{};
  collectUserSpecifiedDDRBanks(F, UserSpecifiedDDRBanks);

  SmallDenseMap<Argument *, unsigned, 16> DDRAssignment{};
  for (auto &Arg : F.args()) {
    if (isArgBuffer(&Arg, SyclHlsFlow)) {
           auto Assignment = getUserSpecifiedDDRBank(&Arg, UserSpecifiedDDRBanks);
           DDRAssignment[&Arg] = (Assignment) ? Assignment.getValue() : 0;
         }
  }

  for (auto &ArgBank : DDRAssignment) {
    auto LookUp = BundlesByIDName.find(ArgBank.second);
    if (LookUp == BundlesByIDName.end()) {
      // We need to create a bundle for this bank
      auto BundleName = formatv("ddrmem{0}", ArgBank.second);
      Bundles.push_back({ BundleName, ArgBank.second});
      unsigned BundleIdx = Bundles.size() - 1;
      BundlesByName[ {BundleName} ] = BundleIdx;
      BundleForArgument[ArgBank.first] = BundleIdx;
      BundlesByIDName[ArgBank.second][ {BundleName} ] = BundleIdx;
    } else {
      BundleForArgument[ArgBank.first] = LookUp->getSecond().begin()->second;
    }
  }
}

 KernelProperties::MAXIBundle const * KernelProperties::getArgumentMAXIBundle(Argument *Arg) {
   auto LookUp = BundleForArgument.find(Arg);
   if (LookUp != BundleForArgument.end()) {
     return &(Bundles[LookUp->second]);
   } 
   return nullptr;
}
} // namespace llvm
