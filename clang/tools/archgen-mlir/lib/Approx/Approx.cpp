//===- Approx.cpp - Approx Dialect ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/OpDefinition.h"

#include "archgen/Approx/Approx.h"

using namespace archgen::approx;

#include "archgen/Approx/ApproxDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ApproxDialect Interfacce
//===----------------------------------------------------------------------===//
struct ApproxInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  /// All call operations should get inlined
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final {
    return true;
  }
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ApproxDialect
//===----------------------------------------------------------------------===//

void ApproxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "archgen/Approx/ApproxOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "archgen/Approx/ApproxType.cpp.inc"
      >();
  addInterfaces<ApproxInlinerInterface>();
}

mlir::Operation *ApproxDialect::materializeConstant(mlir::OpBuilder &builder,
                                                    mlir::Attribute value,
                                                    mlir::Type type,
                                                    mlir::Location loc) {
  assert(type.isa<fixedpt::FixedPtType>());
  return builder.create<approx::ConstantOp>(
      loc, type, value.cast<fixedpt::FixedPointAttr>());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "archgen/Approx/ApproxOps.cpp.inc"

void ConstantOp::build(mlir::OpBuilder &odsBuilder,
                       mlir::OperationState &odsState,
                       archgen::fixedpt::FixedPointAttr value) {
  build(odsBuilder, odsState, toBeFoldedType::get(odsBuilder.getContext()),
        value);
}

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "archgen/Approx/ApproxType.cpp.inc"
