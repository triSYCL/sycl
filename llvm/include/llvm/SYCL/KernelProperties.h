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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {
class KernelProperties {
public:
  struct MAXIBundle {
    std::string BundleName;
    unsigned TargetId;
  };
private:
  // In HLS, array-like arguments are groupped together in bundles.
  // One bundle correspond to one memory controller, and this is 
  // the bundle that can be associated to a specific DDR Bank.
  //
  // As of now, all arguments sharing the same memory bank share the 
  // same bundle.
  SmallDenseMap<unsigned, StringMap<unsigned>, 4> BundlesByIDName;
  SmallDenseMap<Argument *, unsigned, 16> BundleForArgument;
  StringMap<unsigned> BundlesByName;
  SmallVector<MAXIBundle, 8> Bundles;

public:
  static bool isArgBuffer(Argument* Arg, bool SyclHLSFlow);
  KernelProperties(Function &F, bool SyclHlsFlow);
  KernelProperties(KernelProperties &) = delete;

  MAXIBundle const * getArgumentMAXIBundle(Argument *Arg);
  SmallVector<MAXIBundle, 8> const &getMAXIBundles() {
    return Bundles;
  };
};
} // namespace llvm

#endif
