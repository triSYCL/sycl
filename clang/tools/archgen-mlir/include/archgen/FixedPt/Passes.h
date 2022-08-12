//===- Passes.h - Place and route ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declare the Factorio blueprint Pass.
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_FIXEDPT_PASSES_H
#define ARCHGEN_FIXEDPT_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinDialect.h"

namespace archgen {
namespace fixedpt {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertFixedPtToArithPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "archgen/FixedPt/FixedPtPasses.h.inc"

} // namespace fixedpt
} // namespace archgen

#endif
