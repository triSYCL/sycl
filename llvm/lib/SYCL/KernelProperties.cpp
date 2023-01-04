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

#include "llvm/SYCL/SYCLUtils.h"
#include "llvm/SYCL/KernelProperties.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
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

namespace llvm {

KernelProperties::KernelProperties(Function &F) {
  Bundles.push_back(MAXIBundle{{}, "default", sycl::MemoryType::unspecified});

  // For each argument A of F which is a buffer, if it has no user specified
  // assigned Bank, default to "default" bundle
  for (auto &Arg : F.args()) {
    if (sycl::isArgBuffer(&Arg)) {
      auto Assignment = sycl::getMemoryBank(&Arg);
      if (Assignment) {
        auto ArgBank = Assignment;
        auto& SubSpecIndex = BundlesBySpec[static_cast<size_t>(ArgBank.MemType)];
        auto LookUp = SubSpecIndex.find(ArgBank.BankID);
        if (LookUp == SubSpecIndex.end()) {
          // We need to create a bundle for this bank
          StringRef Prefix;
          switch (ArgBank.MemType) {
            case sycl::MemoryType::ddr:
            Prefix = "ddr";
            break;
            case sycl::MemoryType::hbm:
            Prefix = "hb";
            break;
            default:
            llvm_unreachable("Default type should not appear here");
          }
          std::string BundleName{formatv("{0}mem{1}", Prefix, ArgBank.BankID)};
          Bundles.push_back({ArgBank.BankID, BundleName, ArgBank.MemType});
          unsigned BundleIdx = Bundles.size() - 1;
          BundlesByName[BundleName] = BundleIdx;
          BundleForArgument[&Arg] = BundleIdx;
          SubSpecIndex[ArgBank.BankID] = BundleIdx;
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
