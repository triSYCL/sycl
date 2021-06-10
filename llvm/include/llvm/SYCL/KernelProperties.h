//===- KernelProperties.h - Tools for extracting sycl kernel properties ---===//
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

#ifndef LLVM_SYCL_KERNELPROPERTIES_H
#define LLVM_SYCL_KERNELPROPERTIES_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {
class KernelProperties {
private:
  // In HLS, array-like arguments are groupped together in bundles.
  // One bundle correspond to one memory controller, and this is 
  // the bundle that can be associated to a specific DDR Bank.
  //
  // As of now, all arguments sharing the same memory bank share the 
  // same bundle.
  struct MAXIBundle {
    std::string bundleName;
  };
  SmallDenseMap<unsigned, MAXIBundle, 4> maxiBundles;
  SmallDenseMap<AllocaInst *, unsigned, 8> userSpecifiedDDRBanks;

public:
  KernelProperties(Function &F);
  KernelProperties(KernelProperties &orig) = delete;

  Optional<unsigned> getUserSpecifiedDDRBank(Argument *Arg);
  Optional<StringRef> getArgumentMAXIBundle(Argument *Arg);
  SmallDenseMap<unsigned, MAXIBundle, 4> const &getMAXIBundles() {
    return maxiBundles;
  };
};
} // namespace llvm

#endif
