//===- InSPIRation.h - SYCL SPIR fixer pass       -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Rewrite the kernels and functions so that they are compatible with SPIR
// representation as described in "The SPIR Specification Version 2.0 -
// Provisional" from Khronos Group.
// ===---------------------------------------------------------------------===//

#ifndef LLVM_SYCL_INSPIRATION_H
#define LLVM_SYCL_INSPIRATION_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createInSPIRationPass();

}

#endif
