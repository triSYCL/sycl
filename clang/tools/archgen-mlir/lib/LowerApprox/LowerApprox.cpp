//===- LowerApprox.cpp - Replace approx by its aproximation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "archgen/Approx/Approx.h"
#include "archgen/Approx/PassDetail.h"
#include "archgen/Approx/Passes.h"

#include "FixFunctions/BasicPolyApprox.hpp"

using namespace archgen;
using namespace archgen::approx;

namespace {

struct LowerApprox {

  void run(genericOp evaluateOp) {
    
  }
};

struct LowerApproxPass
    : LowerApproxPassBase<LowerApproxPass> {
  virtual void runOnOperation() override final;
};

void LowerApproxPass::runOnOperation() {
  getOperation().walk([&](genericOp op) {
    if (op.action() == action::evaluate)
      LowerApprox{}.run(op);
  });
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::approx::createLowerApproxPass() {
  return std::make_unique<LowerApproxPass>();
}
