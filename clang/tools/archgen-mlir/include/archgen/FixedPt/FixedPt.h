//===- FuncOps.h - Func Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_FIXEDPT_H
#define ARCHGEN_FIXEDPT_H

#include "llvm/ADT/APFixedPoint.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "archgen/FixedPt/FixedPtDialect.h.inc"
#include "archgen/FixedPt/FixedPtEnum.h.inc"

namespace archgen {
namespace fixedpt {

#include "archgen/FixedPt/FixedPtInterfaces.h.inc"

/// Special value to indicate that there is no common rounding mode for a
/// certain combination
constexpr fixedpt::RoundingMode incompatibleRounding =
    static_cast<fixedpt::RoundingMode>(std::numeric_limits<uint32_t>::max());

/// Find a common rounding mode for m1 and m2. if there is none return
/// incompatibleRounding
fixedpt::RoundingMode getCommonRoundingMod(fixedpt::RoundingMode m1,
                                           fixedpt::RoundingMode m2);

/// return true if there is a compatible roundign between m1 and m2
bool hasCommonRounding(fixedpt::RoundingMode m1, fixedpt::RoundingMode m2);

} // namespace fixedpt
} // namespace archgen

#define GET_TYPEDEF_CLASSES
#include "archgen/FixedPt/FixedPtType.h.inc"

#define GET_ATTRDEF_CLASSES
#include "archgen/FixedPt/FixedPtAttr.h.inc"

#define GET_OP_CLASSES
#include "archgen/FixedPt/FixedPtOps.h.inc"

#endif // ARCHGEN_FIXEDPT_H
