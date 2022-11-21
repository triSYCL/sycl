//===- FixedPtPasses.td - Fixed Point passes ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertFixedPtToArithPass : Pass<"convert-fixedpt-to-arith", "::mlir::ModuleOp"> {
  let summary = "convert FixedPt Dialect to Arithmetic Dialect";
  let constructor = "::archgen::fixedpt::createConvertFixedPtToArithPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
  ];
}
