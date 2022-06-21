//===- LowerSYCLMetaData.h ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass to translate high-level sycl optimization hints into low-level hls ones
//
// ===---------------------------------------------------------------------===//

#ifndef LLVM_PREPARE_SYCL_OPT_H
#define LLVM_PREPARE_SYCL_OPT_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class LowerSYCLMetaDataPass : public PassInfoMixin<LowerSYCLMetaDataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

ModulePass *createLowerSYCLMetaDataPass();

}

#endif
