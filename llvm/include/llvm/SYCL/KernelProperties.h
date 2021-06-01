//===- KernelProperties.h - Functions for extracting sycl kernel properties ------===//
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
class KernelProperties {
public:
  KernelProperties(Function &F);
  KernelProperties(KernelProperties &orig) = delete;

  unsigned getUserSpecifiedDDRBank(Argument *Arg);

private:
  SmallDenseMap<AllocaInst *, unsigned, 8> userSpecifiedDDRBanks;
};
} // namespace llvm

#endif