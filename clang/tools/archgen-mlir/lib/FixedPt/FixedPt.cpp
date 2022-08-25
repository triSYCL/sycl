//===- FixedPt.cpp - FixedPt Dialect
//------------------------------------------===//
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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"

#include "archgen/FixedPt/FixedPt.h"

using namespace archgen::fixedpt;
using namespace archgen;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "archgen/FixedPt/FixedPtOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "archgen/FixedPt/FixedPtType.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "archgen/FixedPt/FixedPtAttr.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd interface definitions
//===----------------------------------------------------------------------===//

#include "archgen/FixedPt/FixedPtInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// FixedPtDialect
//===----------------------------------------------------------------------===//

#include "archgen/FixedPt/FixedPtDialect.cpp.inc"

mlir::Operation *FixedPtDialect::materializeConstant(mlir::OpBuilder &builder,
                                                     mlir::Attribute value,
                                                     mlir::Type type,
                                                     mlir::Location loc) {
  return builder.create<fixedpt::ConstantOp>(
      loc, type, value.cast<fixedpt::FixedPointAttr>());
}

void FixedPtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "archgen/FixedPt/FixedPtOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "archgen/FixedPt/FixedPtType.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "archgen/FixedPt/FixedPtAttr.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Fixed Point type definitions
//===----------------------------------------------------------------------===//

mlir::LogicalResult
FixedPtType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    int msb, int lsb, bool is_signed) {
  if (msb < lsb) {
    emitError().append("requires: msb >= lsb:", msb, " >= ", lsb);
    return mlir::failure();
  }
  return mlir::success();
}

llvm::FixedPointSemantics FixedPtType::getFixedPointSemantics() const {
  /// isStaturated and hasUnsignedPadding are not properties of the layout but
  /// properties of operations on the APFixedPoint so they are not represented
  /// by the MLIR type
  return llvm::FixedPointSemantics(
      getWidth(), llvm::FixedPointSemantics::Lsb{getLsb()}, isSigned(),
      /*isStaturated*/ false,
      /*hasUnsignedPadding*/ false);
}

FixedPtType FixedPtType::getCommonMulType(FixedPtType other) const {
  int lsb = getLsb() + other.getLsb();
  int width = getWidth() + other.getWidth();
  return FixedPtType::get(getContext(), width + lsb, lsb,
                          isSigned() || other.isSigned());
}

FixedPtType FixedPtType::getCommonAddType(FixedPtType other) const {
  return FixedPtType::get(getContext(), std::max(getMsb(), other.getMsb()) + 1,
                          std::min(getLsb(), other.getLsb()),
                          isSigned() || other.isSigned());
}

mlir::Type FixedPtType::parse(mlir::AsmParser &odsParser) {
  int msb;
  int lsb;
  std::string sign;
  if (odsParser.parseLess() || odsParser.parseInteger(msb) ||
      odsParser.parseComma() || odsParser.parseInteger(lsb) ||
      odsParser.parseComma() || odsParser.parseString(&sign) ||
      odsParser.parseGreater()) {
    odsParser.emitError(odsParser.getNameLoc(), "failed to parse FixedPtType");
    return {};
  }
  if (sign != "s" && sign != "u") {
    odsParser.emitError(odsParser.getNameLoc(), "expected s or u got " + sign);
  }
  bool isSigned = (sign == "s");
  return FixedPtType::get(odsParser.getContext(), msb, lsb, isSigned);
}

void FixedPtType::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getMsb() << ", " << getLsb() << ", \""
             << (isSigned() ? "s" : "u") << "\">";
}

//===----------------------------------------------------------------------===//
// Fixed Point attribute definitions
//===----------------------------------------------------------------------===//

mlir::Attribute FixedPointAttr::parse(mlir::AsmParser &odsParser,
                                      mlir::Type odsType) {
  mlir::Type ty;
  llvm::APInt rawInt;
  std::string text;
  if (odsParser.parseLess() || odsParser.parseInteger(rawInt) ||
      odsParser.parseComma() || odsParser.parseType(ty) ||
      odsParser.parseComma() || odsParser.parseString(&text) ||
      odsParser.parseGreater()) {
    odsParser.emitError(odsParser.getNameLoc(),
                        "failed to parse FixedPointAttr");
    return {};
  }
  llvm::APInt intPart = rawInt.zextOrTrunc(ty.cast<FixedPtType>().getWidth());
  assert(rawInt == intPart.zextOrTrunc(rawInt.getBitWidth()));
  llvm::APFixedPoint value(intPart,
                           ty.cast<FixedPtType>().getFixedPointSemantics());
  assert(text == value.toString() && "textual value should match");
  return FixedPointAttr::get(odsParser.getContext(), std::move(value));
}

void FixedPointAttr::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getValue().getValue() << ", "
             << FixedPtType::get(this->getContext(), getValue().getSemantics())
             << ", \"" << getValue().toString() << "\""
             << ">";
}
//===----------------------------------------------------------------------===//
// Fixed Point enumerations
//===----------------------------------------------------------------------===//

#include "archgen/FixedPt/FixedPtEnum.cpp.inc"

fixedpt::RoundingMode fixedpt::getCommonRoundingMod(fixedpt::RoundingMode m1,
                                                    fixedpt::RoundingMode m2) {
  assert((unsigned)m1 <= fixedpt::getMaxEnumValForRoundingMode());
  assert((unsigned)m2 <= fixedpt::getMaxEnumValForRoundingMode());
  // clang-format off
  fixedpt::RoundingMode table[] = {
    RoundingMode::zero, RoundingMode::nearest,
    RoundingMode::nearest, RoundingMode::nearest,
  };
  // clang-format on
  auto access = [&](auto x, auto y) {
    return table[(unsigned)m1 * (fixedpt::getMaxEnumValForRoundingMode() + 1) +
                 (unsigned)m2];
  };

#ifndef NDEBUG
  for (unsigned i = 0; i <= fixedpt::getMaxEnumValForRoundingMode(); i++)
    for (unsigned k = 0; k <= fixedpt::getMaxEnumValForRoundingMode(); k++)
      assert(access(i, k) == access(k, i));
#endif
  return access(m1, m2);
}

//===----------------------------------------------------------------------===//
// Fixed Point operation definitions
//===----------------------------------------------------------------------===//

mlir::LogicalResult AddOp::verify() { return mlir::success(); }

mlir::LogicalResult SubOp::verify() { return mlir::success(); }

mlir::LogicalResult MulOp::verify() { return mlir::success(); }

mlir::LogicalResult DivOp::verify() { return mlir::success(); }

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return valueAttr();
}

mlir::LogicalResult ConstantOp::verify() {
  if (getType().getFixedPointSemantics() !=
      valueAttr().getValue().getSemantics())
    return emitError("fixed-point semantic of type doesnt match fixed-point "
                     "semantic of attribute");
  return mlir::success();
}

mlir::LogicalResult TruncOp::verify() {
  FixedPtType inTy = input().getType().cast<FixedPtType>();
  FixedPtType outTy = result().getType().cast<FixedPtType>();
  if (inTy.getLsb() != outTy.getLsb())
    return emitError("cannot change lsb");
  if (inTy.getMsb() <= outTy.getMsb())
    return emitError("the msb must be smaller in the output");
  return mlir::success();
}

mlir::LogicalResult RoundOp::verify() {
  FixedPtType inTy = input().getType().cast<FixedPtType>();
  FixedPtType outTy = result().getType().cast<FixedPtType>();
  if (inTy.getMsb() != outTy.getMsb())
    return emitError("cannot change msb");
  if (inTy.isSigned() != outTy.isSigned())
    return emitError("cannot change sign");
  if (inTy.getLsb() >= outTy.getLsb())
    return emitError("the lsb must be smaller in the input");
  return mlir::success();
}

mlir::LogicalResult ExtendOp::verify() {
  FixedPtType inTy = input().getType().cast<FixedPtType>();
  FixedPtType outTy = result().getType().cast<FixedPtType>();
  if (inTy.getWidth() >= outTy.getWidth())
    return emitError("width must increase");
  if (inTy.getLsb() < outTy.getLsb())
    return emitError("lsb can only decrease");
  if (inTy.getMsb() > outTy.getMsb())
    return emitError("msb can only increase");
  return mlir::success();
}

mlir::LogicalResult BitcastOp::verify() {
  FixedPtType inFPTy = input().getType().dyn_cast<FixedPtType>();
  mlir::IntegerType inIntTy = input().getType().dyn_cast<mlir::IntegerType>();
  FixedPtType outFPTy = result().getType().dyn_cast<FixedPtType>();
  mlir::IntegerType outIntTy = result().getType().dyn_cast<mlir::IntegerType>();
  if (inIntTy && outIntTy)
    return emitError("use arith.bitcast instead");
  if ((inIntTy && inIntTy.getSignedness() != mlir::IntegerType::Signless) ||
      (outIntTy && outIntTy.getSignedness() != mlir::IntegerType::Signless))
    return emitError("integer types must be signless");
  if ((inFPTy && outFPTy && inFPTy.getWidth() != outFPTy.getWidth()) ||
      (inFPTy && outIntTy && inFPTy.getWidth() != outIntTy.getWidth()) ||
      (inIntTy && outFPTy && inIntTy.getWidth() != outFPTy.getWidth()))
    return emitError("bitwidth must match");
  return mlir::success();
}

mlir::LogicalResult ConvertOp::verify() { return mlir::success(); }

//===----------------------------------------------------------------------===//
// Fixed Point operation canonicalization
//===----------------------------------------------------------------------===//

namespace {

bool canUpdateTypeFor(mlir::Value v) {
  /// We can only update the type of v if v comes from an operation.
  return v.getDefiningOp();
}

mlir::LogicalResult tryUpdateType(mlir::Value v, mlir::Type ty,
                                  RoundingMode rounding = RoundingMode::zero) {
  if (canUpdateTypeFor(v)) {
    if (auto inputOp = llvm::dyn_cast<RoundingOpInterface>(v.getDefiningOp()))
      inputOp.setRoundingMode(
          fixedpt::getCommonRoundingMod(rounding, inputOp.getRoundingMode()));
    v.setType(ty);
    return mlir::success();
  }
  return mlir::failure();
}

} // namespace

mlir::LogicalResult AddOp::canonicalize(AddOp op,
                                        mlir::PatternRewriter &rewriter) {
  mlir::Value var;
  fixedpt::FixedPointAttr constant;
  /// var + 0 -> var
  /// I am not sure using the pattern matcher is really easier
  if (((mlir::matchPattern(op.lhs(), mlir::m_Constant(&constant)) &&
        (var = op.rhs())) ||
       (mlir::matchPattern(op.rhs(), mlir::m_Constant(&constant)) &&
        (var = op.lhs()))) &&
      constant.getValue().getValue().isZero()) {
    if (var.getType() == op.result().getType())
      rewriter.replaceOp(op, var);
    else
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.result().getType(), var,
                                             op.rounding());
    return mlir::success();
  }

  /// resize inputs based on outputs
  fixedpt::FixedPtType lhsType = op.lhs().getType().cast<FixedPtType>();
  fixedpt::FixedPtType rhsType = op.rhs().getType().cast<FixedPtType>();
  fixedpt::FixedPtType resType = op.result().getType().cast<FixedPtType>();
  fixedpt::FixedPtType newLhsType = rewriter.getType<FixedPtType>(
      std::min(lhsType.getMsb(), resType.getMsb()),
      std::max(lhsType.getLsb(), resType.getLsb() - 1), lhsType.isSigned());
  fixedpt::FixedPtType newRhsType = rewriter.getType<FixedPtType>(
      std::min(rhsType.getMsb(), resType.getMsb()),
      std::max(rhsType.getLsb(), resType.getLsb() - 1), rhsType.isSigned());

  if (mlir::succeeded(tryUpdateType(op.lhs(), newLhsType, op.rounding())) ||
      mlir::succeeded(tryUpdateType(op.rhs(), newRhsType, op.rounding())))
    return mlir::success();
  return mlir::failure();
}

mlir::LogicalResult MulOp::canonicalize(MulOp op,
                                        mlir::PatternRewriter &rewriter) {
  llvm_unreachable("TODO copy past of add for now");
  fixedpt::FixedPtType lhsType = op.lhs().getType().cast<FixedPtType>();
  fixedpt::FixedPtType rhsType = op.rhs().getType().cast<FixedPtType>();
  fixedpt::FixedPtType resType = op.result().getType().cast<FixedPtType>();
  fixedpt::FixedPtType newLhsType = rewriter.getType<FixedPtType>(
      std::min(lhsType.getMsb(), resType.getMsb()),
      std::max(lhsType.getLsb(), resType.getLsb() - 1), lhsType.isSigned());
  fixedpt::FixedPtType newRhsType = rewriter.getType<FixedPtType>(
      std::min(rhsType.getMsb(), resType.getMsb()),
      std::max(rhsType.getLsb(), resType.getLsb() - 1), rhsType.isSigned());

  if (mlir::succeeded(tryUpdateType(op.lhs(), newLhsType, op.rounding())) ||
      mlir::succeeded(tryUpdateType(op.rhs(), newRhsType, op.rounding())))
    return mlir::success();
  return mlir::failure();
}

mlir::LogicalResult SubOp::canonicalize(SubOp op,
                                        mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult DivOp::canonicalize(DivOp op,
                                        mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult ConstantOp::canonicalize(ConstantOp op,
                                             mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult TruncOp::canonicalize(TruncOp op,
                                          mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult RoundOp::canonicalize(RoundOp op,
                                          mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult ExtendOp::canonicalize(ExtendOp op,
                                           mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult BitcastOp::canonicalize(BitcastOp op,
                                            mlir::PatternRewriter &rewriter) {
  return mlir::failure();
}

mlir::LogicalResult ConvertOp::canonicalize(ConvertOp op,
                                            mlir::PatternRewriter &rewriter) {
  if (mlir::succeeded(
          tryUpdateType(op.input(), op.result().getType(), op.rounding()))) {
    rewriter.replaceOp(op, op.input());
    return mlir::success();
  }
  return mlir::failure();
}
