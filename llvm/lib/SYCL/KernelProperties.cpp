//===- KernelProperties.cpp
//-----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utility functions that read SYCL kernel properties
// ===---------------------------------------------------------------------===//

#include "llvm/SYCL/KernelProperties.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <string>
#include <utility>

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

/// Search for the annotation specifying user specified DDR bank for all
/// arguments of F and populates UserSpecifiedDDRBanks accordingly
void collectUserSpecifiedBanks(
    Function &F,
    SmallDenseMap<llvm::AllocaInst *, KernelProperties::MemBankSpec, 16> &UserSpecifiedBanks) {
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
    KernelProperties::MemoryType MemT;
    if (Str->getRawDataValues() == kindOf("xilinx_ddr_bank"))
      MemT = KernelProperties::MemoryType::DDR;
    else if (Str->getRawDataValues() == kindOf("xilinx_hbm_bank"))
      MemT = KernelProperties::MemoryType::HBM;
    else
      continue;
    Constant *Args =
        (cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(4)))
             ->getInitializer());
    unsigned Bank;
    if (isa<ConstantAggregateZero>(Args))
      Bank = 0;
    else
      Bank = cast<ConstantInt>(Args->getOperand(0))->getZExtValue();

    UserSpecifiedBanks[Alloca] = {MemT, Bank};
  }
}

/// Check if the argument has a user specified DDR bank corresponding to it in
/// UserSpecifiedDDRBank
Optional<KernelProperties::MemBankSpec> getUserSpecifiedBank(
    Argument *Arg,
    SmallDenseMap<llvm::AllocaInst *, KernelProperties::MemBankSpec, 16> &UserSpecifiedBanks) {
  for (User *U : Arg->users()) {
    if (auto *Store = dyn_cast<StoreInst>(U))
      if (Store->getValueOperand() == Arg) {
        auto Lookup = UserSpecifiedBanks.find(dyn_cast_or_null<AllocaInst>(
            getUnderlyingObject(Store->getPointerOperand())));
        if (Lookup == UserSpecifiedBanks.end())
          continue;
        return {Lookup->second};
      }
  }
  return {};
}
} // namespace

namespace llvm {
bool KernelProperties::isArgBuffer(Argument *Arg, bool SyclHLSFlow) {
  if (Arg->getType()->isPointerTy() &&
      (SyclHLSFlow ||
       Arg->getType()->getPointerAddressSpace() == SPIRAS_Global ||
       Arg->getType()->getPointerAddressSpace() == SPIRAS_Constant)) {
    return !Arg->hasByValAttr();
  }
  return false;
}

KernelProperties::KernelProperties(Function &F, bool SyclHlsFlow) {
  Bundles.push_back(MAXIBundle{{}, "default", MemoryType::DEFAULT});
  SmallDenseMap<llvm::AllocaInst *, MemBankSpec, 16> UserSpecifiedBanks{};
  // Collect user specified DDR banks for F in DDRBanks
  collectUserSpecifiedBanks(F, UserSpecifiedBanks);


  // For each argument A of F which is a buffer, if it has no user specified DDR
  // Bank, default to 0
  for (auto &Arg : F.args()) {
    if (isArgBuffer(&Arg, SyclHlsFlow)) {
      auto Assignment = getUserSpecifiedBank(&Arg, UserSpecifiedBanks);
      if (Assignment.hasValue()) {
        auto ArgBank = Assignment.getValue();
        auto& SubSpecIndex = BundlesBySpec[static_cast<size_t>(ArgBank.first)];
        auto LookUp = SubSpecIndex.find(ArgBank.second);
        if (LookUp == SubSpecIndex.end()) {
          // We need to create a bundle for this bank
          StringRef Prefix;
          switch (ArgBank.first) {
            case MemoryType::DDR:
            Prefix = "ddr";
            break;
            case MemoryType::HBM:
            Prefix = "hb";
            break;
            default:
            llvm_unreachable("Default type should not appear here");
          }
          std::string BundleName{formatv("{0}mem{1}", Prefix, ArgBank.second)};
          Bundles.push_back({ArgBank.second, BundleName, ArgBank.first});
          unsigned BundleIdx = Bundles.size() - 1;
          BundlesByName[BundleName] = BundleIdx;
          BundleForArgument[&Arg] = BundleIdx;
          SubSpecIndex[ArgBank.second] = BundleIdx;
        } else {
          BundleForArgument[&Arg] = LookUp->getSecond();
        }
      } else {
        BundleForArgument[&Arg] = 0; // Default Bundle
      }
    }
  }
}

KernelProperties::MAXIBundle const *
KernelProperties::getArgumentMAXIBundle(Argument *Arg) {
  auto LookUp = BundleForArgument.find(Arg);
  if (LookUp != BundleForArgument.end()) {
    return &(Bundles[LookUp->second]);
  }
  return nullptr;
}
} // namespace llvm
