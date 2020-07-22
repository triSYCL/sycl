//===- XOCCIRDowngrader.h - SYCL XOCC IR Downgrader pass  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass for modifying certain LLVM IR incompatabilities with the Xilinx xocc
// backend we use for SYCL
//
// ===---------------------------------------------------------------------===//

#ifndef LLVM_SYCL_XOCC_IR_DOWNGRADER_H
#define LLVM_SYCL_XOCC_IR_DOWNGRADER_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createXOCCIRDowngraderPass();

}

#endif
