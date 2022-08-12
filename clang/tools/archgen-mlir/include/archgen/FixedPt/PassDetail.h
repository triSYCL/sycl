//===- PassDetail.h - FBP Pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_FIXEDPT_PASSDETAIL_H
#define ARCHGEN_FIXEDPT_PASSDETAIL_H

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinDialect.h"
#include "archgen/FixedPt/FixedPt.h"

namespace archgen {
namespace fixedpt {

#define GEN_PASS_CLASSES
#include "archgen/FixedPt/FixedPtPasses.h.inc"

}
}

#endif // ARCHGEN_FIXEDPT_PASSDETAIL_H
