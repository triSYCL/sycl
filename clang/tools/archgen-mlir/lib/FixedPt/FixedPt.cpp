//===- FixedPt.cpp - FixedPt Dialect
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>

#include "llvm/ADT/APFixedPoint.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "archgen/FixedPt/FixedPt.h"

using namespace archgen::fixedpt;
using namespace archgen;

namespace arith = mlir::arith;

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
  int w1 = getWidth();
  int w2 = other.getWidth();
  unsigned int generalProdWidth = w1 + w2;
  unsigned int max = (w1 > w2) ? w1 : w2;
  bool sameSignedness = (isSigned() == other.isSigned());
  bool oneIsOne = ((w1 == 1) || (w2 == 1));
  bool bothAreOne = ((w1 == 1) && (w2 == 1));
  unsigned int caseOneWidth = (sameSignedness || bothAreOne) ? max : max + 1;
  bool oneSigned = isSigned() or other.isSigned();
  bool prodSigned = (oneSigned and (!bothAreOne || !sameSignedness));
  unsigned int prodSize = (oneIsOne) ? caseOneWidth : generalProdWidth;

  int lsb = getLsb() + other.getLsb();
  return FixedPtType::get(getContext(), prodSize + lsb - 1, lsb, prodSigned);
}

FixedPtType FixedPtType::getCommonAddType(FixedPtType other) const {
  auto lsb_out = std::min(getLsb(), other.getLsb());
  auto max_msb = std::max(getMsb(), other.getMsb());
  auto max_pos_msb = std::max(getMaxPositiveBW(), other.getMaxPositiveBW());
  auto one_signed = isSigned() || other.isSigned();
  auto msb =
      1 + (((max_msb == max_pos_msb) && one_signed) ? max_msb + 1 : max_msb);
  return FixedPtType::get(getContext(), msb, lsb_out, one_signed);
}

mlir::Type FixedPtType::parse(mlir::AsmParser &odsParser) {
  int msb;
  int lsb;
  llvm::StringRef signKW;
  if (odsParser.parseLess() || odsParser.parseInteger(msb) ||
      odsParser.parseComma() || odsParser.parseInteger(lsb) ||
      odsParser.parseComma() || odsParser.parseKeyword(&signKW) ||
      odsParser.parseGreater()) {
    odsParser.emitError(odsParser.getNameLoc(), "failed to parse FixedPtType");
    return {};
  }
  if (signKW != "s" && signKW != "u") {
    odsParser.emitError(odsParser.getNameLoc(),
                        "expected s or u got " + signKW);
  }
  bool isSigned = (signKW == "s");
  return FixedPtType::get(odsParser.getContext(), msb, lsb, isSigned);
}

void FixedPtType::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getMsb() << ", " << getLsb() << ", "
             << (isSigned() ? "s" : "u") << ">";
}

//===----------------------------------------------------------------------===//
// Fixed Point attribute definitions
//===----------------------------------------------------------------------===//

mlir::Attribute FixedPointAttr::parse(mlir::AsmParser &odsParser,
                                      mlir::Type odsType) {
  FixedPtType ty;
  llvm::SMLoc intLoc;
  llvm::APInt rawInt;
  std::string text;
  llvm::SMLoc textLoc;
  if (odsParser.parseLess() || odsParser.getCurrentLocation(&intLoc) ||
      odsParser.parseInteger(rawInt) || odsParser.parseComma() ||
      odsParser.parseCustomTypeWithFallback(ty) || odsParser.parseComma() ||
      odsParser.getCurrentLocation(&textLoc) || odsParser.parseString(&text) ||
      odsParser.parseGreater()) {
    odsParser.emitError(odsParser.getNameLoc(),
                        "failed to parse FixedPointAttr");
    return {};
  }
  llvm::APInt intPart = rawInt.zextOrTrunc(ty.cast<FixedPtType>().getWidth());
  if (rawInt != intPart.zextOrTrunc(rawInt.getBitWidth())) {
    odsParser.emitError(intLoc, "integer doesn't fit in format");
    return {};
  }
  llvm::APFixedPoint value(intPart,
                           ty.cast<FixedPtType>().getFixedPointSemantics());
  if (text != value.toString()) {
    odsParser.emitError(textLoc, "textual value is incorrect");
    return {};
  }
  return FixedPointAttr::get(odsParser.getContext(), std::move(value));
}

void FixedPointAttr::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getValue().getValue() << ", ";
  odsPrinter.printStrippedAttrOrType(
      FixedPtType::get(this->getContext(), getValue().getSemantics()));
  odsPrinter << ", \"" << getValue().toString() << "\""
             << ">";
}

//===----------------------------------------------------------------------===//
// Fixed Point enumerations
//===----------------------------------------------------------------------===//

#include "archgen/FixedPt/FixedPtEnum.cpp.inc"

namespace {

auto lookupCommonRoundingRaw = [](auto x, auto y) {
  // clang-format off
  static fixedpt::RoundingMode commonRoundingTable[] = {
    RoundingMode::truncate, RoundingMode::nearest, RoundingMode::nearest_even_to_up, RoundingMode::nearest_even_to_down, incompatibleRounding,
    RoundingMode::nearest, RoundingMode::nearest, RoundingMode::nearest_even_to_up, RoundingMode::nearest_even_to_down, incompatibleRounding,
    RoundingMode::nearest_even_to_up, RoundingMode::nearest_even_to_up, incompatibleRounding, incompatibleRounding, incompatibleRounding,
    RoundingMode::nearest_even_to_down, RoundingMode::nearest_even_to_down, incompatibleRounding, incompatibleRounding, incompatibleRounding,
    incompatibleRounding, incompatibleRounding, incompatibleRounding, incompatibleRounding, incompatibleRounding,
  };
  // clang-format on
  return commonRoundingTable[(unsigned)x *
                                 (fixedpt::getMaxEnumValForRoundingMode() + 2) +
                             (unsigned)y];
};

bool isValidOrIncompatibleRounding(fixedpt::RoundingMode m) {
  return ((unsigned)m <= fixedpt::getMaxEnumValForRoundingMode()) ||
         m == incompatibleRounding;
}

} // namespace

bool fixedpt::hasCommonRounding(fixedpt::RoundingMode m1, fixedpt::RoundingMode m2) {
  assert(isValidOrIncompatibleRounding(m1));
  assert(isValidOrIncompatibleRounding(m2));
  return lookupCommonRoundingRaw(m1, m2) != incompatibleRounding;
}

fixedpt::RoundingMode fixedpt::getCommonRoundingMod(fixedpt::RoundingMode m1,
                                                    fixedpt::RoundingMode m2) {
  assert(isValidOrIncompatibleRounding(m1));
  assert(isValidOrIncompatibleRounding(m2));

#ifndef NDEBUG
  /// common<A, B> == common<B, A>, so the table should be symmetric
  /// This code makes sure it is.
  for (unsigned i = 0; i <= fixedpt::getMaxEnumValForRoundingMode(); i++)
    for (unsigned k = 0; k <= fixedpt::getMaxEnumValForRoundingMode(); k++)
      assert(lookupCommonRoundingRaw(i, k) == lookupCommonRoundingRaw(k, i));
#endif
  return lookupCommonRoundingRaw(m1, m2);
}

//===----------------------------------------------------------------------===//
// Fixed Point operation definitions
//===----------------------------------------------------------------------===//

namespace {

void printVariadicOp(mlir::Operation *op, mlir::OpAsmPrinter &printer) {
  bool isFirst = true;
  printer << " ";
  for (mlir::Value v : op->getOperands()) {
    printer << (isFirst ? (isFirst = false, "") : ", ") << v << " : ";
    printer.printStrippedAttrOrType(v.getType().cast<FixedPtType>());
  }
  printer << " "
          << ConvertToString(
                 op->getAttrOfType<RoundingModeAttr>("rounding").getValue())
          << " ";
  printer.printStrippedAttrOrType(
      op->getResultTypes().front().cast<FixedPtType>());
  printer.printOptionalAttrDict(op->getAttrs(), {"rounding"});
}

mlir::ParseResult parseVariadicOp(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  llvm::StringRef roundingStr;
  llvm::SMLoc roundKWLoc;
  fixedpt::FixedPtType resultTy;
  do {
    mlir::OpAsmParser::UnresolvedOperand operand;
    fixedpt::FixedPtType ty;
    if (parser.parseOperand(operand) || parser.parseColon() ||
        parser.parseCustomTypeWithFallback(ty) ||
        parser.resolveOperand(operand, ty, result.operands))
      return mlir::failure();
  } while (!parser.parseOptionalComma());
  if (parser.getCurrentLocation(&roundKWLoc) ||
      parser.parseKeyword(&roundingStr) ||
      parser.parseCustomTypeWithFallback(resultTy) ||
      parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  result.types.push_back(resultTy);
  auto rounding = ConvertToEnum(roundingStr);
  if (!rounding)
    return parser.emitError(roundKWLoc, "expected rounding mode");
  result.attributes.append(
      "rounding",
      RoundingModeAttr::get(parser.getContext(), *ConvertToEnum(roundingStr)));
  return mlir::success();
}

} // namespace

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseVariadicOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) {
  printVariadicOp(this->getOperation(), p);
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseVariadicOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) {
  printVariadicOp(this->getOperation(), p);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  FixedPtType ty;
  llvm::SMLoc intLoc = parser.getCurrentLocation();
  llvm::APInt rawInt;
  std::string text;
  llvm::SMLoc textLoc;

  if (parser.parseInteger(rawInt) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(ty)  || parser.parseComma() ||
      parser.getCurrentLocation(&textLoc) || parser.parseString(&text))
    return parser.emitError(parser.getNameLoc(),
                            "failed to parse fixedpt.constant");

  llvm::APSInt rawSInt(rawInt, ty.isUnsigned());
  llvm::APSInt intPart = rawSInt.extOrTrunc(ty.getWidth());
  if (rawInt != intPart.extOrTrunc(rawInt.getBitWidth()))
    return parser.emitError(intLoc, "integer doesn't fit in format");
  llvm::APFixedPoint value(intPart,
                           ty.getFixedPointSemantics());
  if (text != value.toString())
    return parser.emitError(textLoc, "expected: ").append(value.toString());
  result.addAttribute(
      ConstantOp::getAttributeNames()[0],
      FixedPointAttr::get(parser.getContext(), std::move(value)));
  result.types.push_back(ty);

  return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &p) {
  p << " " << valueAttr().getValue().getValue() << " : ";
  p.printStrippedAttrOrType(result().getType().cast<FixedPtType>());
  p << ", \"" << valueAttr().getValue().toString() << "\"";
}

void ConstantOp::build(mlir::OpBuilder &odsBuilder,
                       mlir::OperationState &odsState,
                       fixedpt::FixedPointAttr val) {
  ConstantOp::build(odsBuilder, odsState,
                    fixedpt::FixedPtType::get(odsBuilder.getContext(),
                                              val.getValue().getSemantics()),
                    val);
}
void ConstantOp::build(mlir::OpBuilder &odsBuilder,
                       mlir::OperationState &odsState, llvm::APFixedPoint val) {
  ConstantOp::build(odsBuilder, odsState,
                    fixedpt::FixedPointAttr::get(odsState.getContext(), val));
}

mlir::LogicalResult AddOp::verify() {
  if (args().size() < 2)
    return emitError().append("requires at least two operands");
  return mlir::success();
}

mlir::LogicalResult SubOp::verify() { return mlir::success(); }

mlir::LogicalResult MulOp::verify() {
  if (args().size() < 2)
    return emitError().append("requires at least two operands");
  return mlir::success();
}

mlir::LogicalResult DivOp::verify() { return mlir::success(); }

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return valueAttr();
}

mlir::LogicalResult ConstantOp::verify() {
  if (getType().getFixedPointSemantics() !=
      valueAttr().getValue().getSemantics())
    return emitError("fixed-point semantic of type doesn't match fixed-point "
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

llvm::APFixedPoint exactAdd(llvm::APFixedPoint const &lhs,
                            llvm::APFixedPoint const &rhs) {
  // We only need to extend one of the values to the suitable msb, as required
  // LSB extension is done within APFixedPoint add
  auto LsbOut = lhs.getLsbWeight();
  auto MaxMsb = std::max(lhs.getMsbWeight(), rhs.getMsbWeight());
  auto MaxPosMsb = std::max(lhs.getMaxPosWeight(), rhs.getMaxPosWeight());
  auto OneIsSigned = lhs.isSigned() || rhs.isSigned();
  auto MsbOut =
      1 + (((MaxMsb == MaxPosMsb) && OneIsSigned) ? MaxMsb + 1 : MaxMsb);
  assert(MsbOut >= LsbOut);
  unsigned int ResWidth{static_cast<unsigned int>(MsbOut - LsbOut + 1)};
  llvm::FixedPointSemantics ExtendedLHSSema{
      ResWidth, llvm::FixedPointSemantics::Lsb{LsbOut}, OneIsSigned, false,
      false};
  bool overflow;
  auto ext_lhs = lhs.convert(ExtendedLHSSema, &overflow);
  assert(!overflow);
  auto res = ext_lhs.add(rhs, &overflow);
  assert(!overflow);
  return res;
}

llvm::APFixedPoint exactMul(llvm::APFixedPoint const &lhs,
                            llvm::APFixedPoint const &rhs) {
  /// Find a common semantics such that lhs, rhs and the result of there product
  /// fit without overflow or loss of precision.
  llvm::FixedPointSemantics commonSema =
      lhs.getSemantics().getCommonSemantics(rhs.getSemantics());
  int lsb = std::min(std::min(lhs.getLsbWeight(), rhs.getLsbWeight()),
                     lhs.getLsbWeight() + rhs.getLsbWeight());
  int msb = std::max(std::max(lhs.getMsbWeight(), rhs.getMsbWeight()),
                     lhs.getMsbWeight() + rhs.getMsbWeight());
  unsigned width = msb - lsb + 1 + commonSema.isSigned();
  commonSema = llvm::FixedPointSemantics{
      width, llvm::FixedPointSemantics::Lsb{lsb}, commonSema.isSigned(),
      commonSema.isSaturated(), commonSema.hasUnsignedPadding()};

  bool overflow = false;
  auto lhsExt = lhs.convert(commonSema, &overflow);
  assert(!overflow && lhs == lhsExt);
  auto rhsExt = rhs.convert(commonSema, &overflow);
  assert(!overflow && rhs == rhsExt);
  auto res = lhsExt.mul(rhsExt, &overflow);
  assert(!overflow);
  return res;
}

/// Remove leading an trailing bit from a fixedpoint constant.
/// It is not applied as a pattern on every constant because we fold convert of
/// constant to constant. but is it useful as part of some patterns.
fixedpt::FixedPointAttr fitConstantBitwidth(fixedpt::FixedPointAttr ConstVal) {
  auto ConstAPFixed = ConstVal.getValue();
  auto InitMsb = ConstAPFixed.getMsbWeight();
  auto InitLsb = ConstAPFixed.getLsbWeight();
  auto InitIsSigned = ConstAPFixed.isSigned();

  auto ConstAPInt = ConstAPFixed.getValue();

  bool NewSign = ConstAPInt.isNegative();
  auto CLZ = ConstAPInt.countLeadingZeros();
  auto CLO = ConstAPInt.countLeadingOnes();
  auto CTZ = ConstAPInt.countTrailingZeros();

  // Remove trailing zero bits
  int NewLsb = ConstAPInt.isZero() ? 0 : (InitLsb + CTZ);
  auto UselessTopBits = NewSign ? CLO - 1 : CLZ;
  int NewMsb = ConstAPInt.isZero() ? 1 : InitMsb - UselessTopBits;

  if (NewLsb == InitLsb && NewMsb == InitMsb && NewSign == InitIsSigned)
    return ConstVal;

  assert(NewMsb >= NewLsb);

  /// fixedpt.fixedPt<0, 0, *> is not legal
  if (NewMsb == NewLsb)
    NewMsb++;

  ConstAPInt.setIsSigned(NewSign);

  if (!ConstAPInt.isZero())
    ConstAPInt.ashrInPlace(CTZ);

  unsigned int NewWidth{static_cast<unsigned int>(NewMsb - NewLsb) + 1};

  auto NewConstRepr = ConstAPInt.trunc(NewWidth);

  llvm::FixedPointSemantics NewFormat{
      NewWidth, llvm::FixedPointSemantics::Lsb{NewLsb}, NewSign, false, false};
  llvm::APFixedPoint NewCstFix{NewConstRepr, NewFormat};
  return fixedpt::FixedPointAttr::get(ConstVal.getContext(),
                                      std::move(NewCstFix));
}

template <typename OpT>
struct MergeVariadicCommutativeOp : public mlir::RewritePattern {
  MergeVariadicCommutativeOp(mlir::MLIRContext *context)
      : RewritePattern(OpT::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *rawOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto op = llvm::dyn_cast<OpT>(rawOp);
    if (!op)
      return mlir::failure();
    assert(rawOp->hasTrait<mlir::OpTrait::IsCommutative>());
    bool needsReplace = false;
    llvm::SmallVector<mlir::Value> newOperands;
    for (mlir::Value v : rawOp->getOperands())
      if (auto opAbove = v.getDefiningOp<OpT>()) {
        needsReplace = true;
        newOperands.append(opAbove->operand_begin(), opAbove->operand_end());
      } else
        newOperands.push_back(v);
    if (!needsReplace)
      return mlir::failure();

    rewriter.updateRootInPlace(rawOp, [&] { rawOp->setOperands(newOperands); });
    return mlir::success();
  }
};

struct AddOpConstFolder : public mlir::RewritePattern {
  AddOpConstFolder(mlir::MLIRContext *context)
      : RewritePattern(AddOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    AddOp addOp = llvm::cast<AddOp>(op);
    fixedpt::FixedPointAttr constVal;
    llvm::APFixedPoint CstSumVal{
        0, llvm::FixedPointSemantics(1, 0, false, false, false)};
    mlir::SmallVector<mlir::Value> NewArgs;
    int nbCst = 0;
    for (auto arg : addOp.args()) {
      if (mlir::matchPattern(arg, mlir::m_Constant(&constVal))) {
        auto CstValue = constVal.getValue();
        nbCst++;
        CstSumVal = exactAdd(CstSumVal, CstValue);
      } else
        NewArgs.push_back(arg);
    }
    if (nbCst == 0 || (nbCst == 1 && !CstSumVal.getValue().isZero()))
      return mlir::failure();

    if (!CstSumVal.getValue().isZero() || NewArgs.size() == 0) {
      // We need to create a constant op of the resulting constant
      auto CstAttr = fitConstantBitwidth(
          fixedpt::FixedPointAttr::get(getContext(), CstSumVal));
      NewArgs.push_back(
          rewriter.create<fixedpt::ConstantOp>(op->getLoc(), CstAttr));
    }
    assert(NewArgs.size() >= 1);
    if (NewArgs.size() == 1)
      rewriter.replaceOpWithNewOp<fixedpt::ConvertOp>(
          op, addOp.getResult().getType().cast<fixedpt::FixedPtType>(),
          NewArgs[0], addOp.rounding());
    else
      rewriter.replaceOpWithNewOp<fixedpt::AddOp>(
          op, addOp.getResult().getType().cast<fixedpt::FixedPtType>(),
          addOp.rounding(), NewArgs);
    return mlir::success();
  }
};

struct MulOpConstFolder : public mlir::RewritePattern {
  MulOpConstFolder(mlir::MLIRContext *context)
      : RewritePattern(MulOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    MulOp mulOp = llvm::cast<MulOp>(op);
    fixedpt::FixedPointAttr constVal;
    llvm::APFixedPoint fixedPtOne{
        1, llvm::FixedPointSemantics(1, 0, false, false, false)};
    llvm::APFixedPoint cstProdVal = fixedPtOne;
    mlir::SmallVector<mlir::Value> NewArgs;
    int nbCst = 0;
    for (auto arg : op->getOperands())
      if (mlir::matchPattern(arg, mlir::m_Constant(&constVal))) {
        auto cstValue = constVal.getValue();
        /// Any multiplication by 0 result in 0 so fast exit.
        if (cstValue.getValue().isZero()) {
          rewriter.replaceOpWithNewOp<fixedpt::ConstantOp>(
              op, llvm::APFixedPoint{0, mulOp.result()
                                            .getType()
                                            .cast<FixedPtType>()
                                            .getFixedPointSemantics()});
          return mlir::success();
        }
        cstProdVal = exactMul(cstProdVal, cstValue);
        nbCst++;
      } else
        NewArgs.push_back(arg);

    /// If op is already canonical dont do anything.
    if (nbCst == 0 || (nbCst == 1 && cstProdVal != fixedPtOne))
      return mlir::failure();

    if (cstProdVal != fixedPtOne || NewArgs.size() == 0) {
      // We need to create a constant op of the resulting constant
      auto CstAttr = fitConstantBitwidth(
          fixedpt::FixedPointAttr::get(getContext(), cstProdVal));
      NewArgs.push_back(
          rewriter.create<fixedpt::ConstantOp>(op->getLoc(), CstAttr));
    }

    assert(NewArgs.size() >= 1);
    if (NewArgs.size() == 1)
      rewriter.replaceOpWithNewOp<fixedpt::ConvertOp>(
          op, mulOp.getResult().getType().cast<fixedpt::FixedPtType>(),
          NewArgs[0], mulOp.rounding());
    else
      rewriter.replaceOpWithNewOp<fixedpt::MulOp>(
          op, mulOp.getResult().getType().cast<fixedpt::FixedPtType>(),
          mulOp.rounding(), NewArgs);
    return mlir::success();
  }
};

} // namespace

void MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<MulOpConstFolder, MergeVariadicCommutativeOp<MulOp>>(context);
}

void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<AddOpConstFolder, MergeVariadicCommutativeOp<AddOp>>(context);
}

mlir::LogicalResult MulOp::canonicalize(MulOp op,
                                        mlir::PatternRewriter &rewriter) {
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
  mlir::Type outTy = op.result().getType();
  mlir::Type inTy = op.input().getType();

  /// Same type so we remove the bitcast
  if (inTy == outTy) {
    rewriter.replaceOp(op, op.input());
    return mlir::success();
  }
  mlir::Operation *opAbove = op.input().getDefiningOp();
  if (!opAbove)
    return mlir::failure();

  /// bitcast of bitcast we can fold it to 1 bitcast (maybe arith.bitcast)
  if (auto bc = llvm::dyn_cast<fixedpt::BitcastOp>(opAbove)) {
    mlir::Type inAboveTy = bc.input().getType();
    if (inAboveTy == outTy) {
      rewriter.replaceOp(op, bc->getOperands());
      return mlir::success();
    }
    rewriter.replaceOpWithNewOp<fixedpt::BitcastOp>(op, outTy, bc.input());
    return mlir::success();
  }
  return mlir::failure();
}

mlir::LogicalResult ConvertOp::canonicalize(ConvertOp op,
                                            mlir::PatternRewriter &rewriter) {
  FixedPtType outTy = op.result().getType().cast<FixedPtType>();
  if (op.input().getType() == outTy) {
    rewriter.replaceOp(op, op.input());
    return mlir::success();
  }
  mlir::Operation *opAbove = op.input().getDefiningOp();
  if (!opAbove)
    return mlir::failure();

  if (ConstantOp cstOp = llvm::dyn_cast<fixedpt::ConstantOp>(opAbove)) {
    llvm::APFixedPoint newVal =
        cstOp.valueAttr().getValue().convert(outTy.getFixedPointSemantics());
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, outTy,
        FixedPointAttr::get(rewriter.getContext(), std::move(newVal)));
    return mlir::success();
  }

  fixedpt::RoundingMode roundingAbove = [&] {
    if (auto RO = llvm::dyn_cast<RoundingOpInterface>(opAbove))
      return RO.getRoundingMode();
    return fixedpt::RoundingMode::truncate;
  }();

  if (llvm::isa<AddOp, MulOp, DivOp, SubOp>(opAbove) && roundingAbove == op.rounding()) {
    RoundingOpInterface newOp = rewriter.clone(*opAbove);
    newOp->getResult(0).setType(op.result().getType());
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
  if (llvm::isa<TruncOp, ExtendOp>(opAbove) ||
      (llvm::isa<RoundOp, ConvertOp>(opAbove) &&
       roundingAbove == op.rounding())) {
    rewriter.updateRootInPlace(op,
                               [&] { op.setOperand(opAbove->getOperand(0)); });
    return mlir::success();
  }
  return mlir::failure();
}
