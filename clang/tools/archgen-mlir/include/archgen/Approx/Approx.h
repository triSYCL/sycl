//===- FuncOps.h - Func Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_APROX_DILECT_H
#define ARCHGEN_APROX_DILECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "archgen/FixedPt/FixedPt.h"

#include "archgen/Approx/ApproxDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "archgen/Approx/ApproxType.h.inc"

#define GET_OP_CLASSES
#include "archgen/Approx/ApproxOps.h.inc"

#endif // ARCHGEN_APROX_DILECT_H
