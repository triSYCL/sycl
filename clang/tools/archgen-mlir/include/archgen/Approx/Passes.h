//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_APPROX_PASSES_H
#define ARCHGEN_APPROX_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinDialect.h"

namespace archgen {
namespace approx {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerApproxPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "archgen/Approx/ApproxPasses.h.inc"

} // namespace approx
} // namespace archgen

#endif
