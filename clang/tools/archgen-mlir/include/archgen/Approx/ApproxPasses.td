//===- FixedPtPasses.td - Fixed Point passes ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def LowerApproxPass : Pass<"lower-approx", "::mlir::ModuleOp"> {
  let summary = "replace a description of the approximation by the actual approximation";
  let constructor = "::archgen::approx::createLowerApproxPass()";
  let dependentDialects = [
    "::mlir::arith::ArithmeticDialect",
    "::archgen::fixedpt::FixedPtDialect",
  ];
}
