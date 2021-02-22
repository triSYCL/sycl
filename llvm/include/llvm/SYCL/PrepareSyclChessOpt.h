//===- PrepareSyclChessOpt.h ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Prepare sycl chess device code for optimizations
//
// ===---------------------------------------------------------------------===//

#ifndef LLVM_PREPARE_SYCL_OPT_H
#define LLVM_PREPARE_SYCL_OPT_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createPrepareSyclChessOptPass();

}

#endif
