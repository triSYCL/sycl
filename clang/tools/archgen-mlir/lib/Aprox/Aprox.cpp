//===- Aprox.cpp - Aprox Dialect ------------------------------------------===//
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

#include "archgen/Aprox/Aprox.h"

using namespace archgen::aprox;

#include "archgen/Aprox/AproxDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AproxDialect Interfacce
//===----------------------------------------------------------------------===//
struct AproxInlinerInterface : public mlir::DialectInlinerInterface {
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
// AproxDialect
//===----------------------------------------------------------------------===//

void AproxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "archgen/Aprox/AproxOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "archgen/Aprox/AproxType.cpp.inc"
      >();
  addInterfaces<AproxInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "archgen/Aprox/AproxOps.cpp.inc"

void constantOp::build(mlir::OpBuilder &odsBuilder,
                       mlir::OperationState &odsState,
                       archgen::fixedpt::fixedPointAttr value) {
  build(odsBuilder, odsState, toBeFoldedType::get(odsBuilder.getContext()),
        value);
}

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "archgen/Aprox/AproxType.cpp.inc"
