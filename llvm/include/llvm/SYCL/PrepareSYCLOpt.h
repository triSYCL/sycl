//===- PrepareSYCLOpt.h --------------------------------------------------===//
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

#ifndef LLVM_SYCL_PREPARE_SYCL_OPT_H
#define LLVM_SYCL_PREPARE_SYCL_OPT_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class PrepareSYCLOptPass : public PassInfoMixin<PrepareSYCLOptPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createPrepareSYCLOptLegacyPass();

}

#endif
