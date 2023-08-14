//===- archgen-lsp-server ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "archgen/Approx/Approx.h"
#include "archgen/FixedPt/FixedPt.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<archgen::approx::ApproxDialect>();
  registry.insert<archgen::fixedpt::FixedPtDialect>();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
