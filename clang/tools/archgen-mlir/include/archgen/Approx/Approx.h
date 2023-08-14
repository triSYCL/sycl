//===- Approx.h - Approx Dialect Operations ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_APPROX_DILECT_H
#define ARCHGEN_APPROX_DILECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "archgen/FixedPt/FixedPt.h"

#include "archgen/Approx/ApproxEnum.h.inc"

#include "archgen/Approx/ApproxDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "archgen/Approx/ApproxType.h.inc"

#define GET_OP_CLASSES
#include "archgen/Approx/ApproxOps.h.inc"

#endif // ARCHGEN_APPROX_DILECT_H
