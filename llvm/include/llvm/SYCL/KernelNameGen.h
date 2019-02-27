//===- KernelNameGen.h - SYCL Kernel Name Gen pass  -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Retrieves the names of the kernels inside of the passed in file and places
// them into a text file. Possible to merge this into another pass if
// required, as it's a fairly trivial pass on its own.
// ===---------------------------------------------------------------------===//

#ifndef LLVM_SYCL_KERNEL_NAME_GEN_H
#define LLVM_SYCL_KERNEL_NAME_GEN_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createKernelNameGenPass();

}

#endif
