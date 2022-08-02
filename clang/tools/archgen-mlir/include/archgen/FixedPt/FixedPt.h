//===- FuncOps.h - Func Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_FIXEDPT_DILECT_H
#define ARCHGEN_FIXEDPT_DILECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/APFixedPoint.h"

#include "archgen/FixedPt/FixedPtDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "archgen/FixedPt/FixedPtType.h.inc"

#define GET_ATTRDEF_CLASSES
#include "archgen/FixedPt/FixedPtAttr.h.inc"

#define GET_OP_CLASSES
#include "archgen/FixedPt/FixedPtOps.h.inc"

#endif // ARCHGEN_FIXEDPT_DILECT_H
