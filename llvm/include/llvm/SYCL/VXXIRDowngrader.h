//===- VXXIRDowngrader.h - SYCL V++ IR Downgrader pass  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass for modifying certain LLVM IR incompatabilities with the Xilinx v++
// backend we use for SYCL
//
// ===---------------------------------------------------------------------===//

#ifndef LLVM_SYCL_VXX_IR_DOWNGRADER_H
#define LLVM_SYCL_VXX_IR_DOWNGRADER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class VXXIRDowngraderPass : public PassInfoMixin<VXXIRDowngraderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createVXXIRDowngraderLegacyPass();

}

#endif
