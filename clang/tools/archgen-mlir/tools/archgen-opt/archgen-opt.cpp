//===- archgen-opt.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "archgen/Approx/Approx.h"
#include "archgen/FixedPt/FixedPt.h"
#include "archgen/FixedPt/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  archgen::fixedpt::registerPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<archgen::approx::ApproxDialect>();
  registry.insert<archgen::fixedpt::FixedPtDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ArchGen opt tool\n", registry));
}
