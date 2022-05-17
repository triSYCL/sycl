//===- KernelProperties.h - Tools for extracting SYCL kernel properties ---===//
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
  // Regroup SYCL kernel properties that can be of use for downstream tools
  // currently, retrieve annotation for DDR bank assignment to kernel arguments
public:
  // In HLS, array-like arguments are grouped together in bundles.
  // One bundle corresponds to one memory controller, and this is
  // the bundle that can be associated to a specific DDR Bank/HBM.
  //
  // As of now, all arguments sharing the same memory bank share the 
  // same bundle.

  enum struct MemoryType {
      unspecified,
      ddr,
      hbm
  };

  struct MemBankSpec {
    MemoryType MemType;
    unsigned BankID;
  };

  struct MAXIBundle {
    // Represents one m_axi bundle and its associated memory bank ID and Type.
    Optional<unsigned> TargetId; // Associated bank ID
    std::string BundleName; // Vitis bundle name
    MemoryType MemType;
    bool isDefaultBundle() const {
      return MemType == MemoryType::unspecified;
    }
  };

private:
  
  // BundlesBySpec[MemType][MemID] contains the index of the 
  // Bundle of the bank MemType:MemID in Bundles
  std::array<SmallDenseMap<unsigned, unsigned, 4>, 3> BundlesBySpec;
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
