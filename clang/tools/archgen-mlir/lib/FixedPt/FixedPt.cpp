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

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  p << " " << getValueAttr().getValue().getValue() << " : ";
  p.printStrippedAttrOrType(getResult().getType().cast<FixedPtType>());
  p << ", \"" << getValueAttr().getValue().toString() << "\"";
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
  if (getArgs().size() < 2)
    return emitError().append("requires at least two operands");
  return mlir::success();
}

mlir::LogicalResult SubOp::verify() { return mlir::success(); }

mlir::LogicalResult MulOp::verify() {
  if (getArgs().size() < 2)
    return emitError().append("requires at least two operands");
  return mlir::success();
}

mlir::LogicalResult DivOp::verify() { return mlir::success(); }

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return getValueAttr();
}

mlir::LogicalResult ConstantOp::verify() {
  if (getType().getFixedPointSemantics() !=
      getValueAttr().getValue().getSemantics())
    return emitError("fixed-point semantic of type doesn't match fixed-point "
                     "semantic of attribute");
  return mlir::success();
}

mlir::LogicalResult SelectOp::verify() {
  auto inType = getKey().getType().dyn_cast<FixedPtType>();
  auto width = inType.getWidth();
  auto operands = getValues();
  if (operands.size() != 1 << width)
    return emitError("Number of values does not match key size");
  auto frontType = operands.front().getType();
  for (auto t : operands.getTypes()) {
    if (t != frontType)
      return emitError("Select op does not accept mixed values");
  }
  if (getResult().getType() != frontType)
    return emitError("Return type does not match element types");
  return mlir::success();
}

mlir::LogicalResult BitcastOp::verify() {
  FixedPtType inFPTy = getInput().getType().dyn_cast<FixedPtType>();
  mlir::IntegerType inIntTy = getInput().getType().dyn_cast<mlir::IntegerType>();
  FixedPtType outFPTy = getResult().getType().dyn_cast<FixedPtType>();
  mlir::IntegerType outIntTy = getResult().getType().dyn_cast<mlir::IntegerType>();
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

mlir::OpFoldResult foldVariadicCommutative(
    mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands,
    llvm::APFixedPoint identity,
    llvm::function_ref<llvm::APFixedPoint(llvm::APFixedPoint,
                                          llvm::APFixedPoint)>
        calculation,
    llvm::function_ref<mlir::OpFoldResult(
        llvm::SmallVectorImpl<mlir::Value> &newArgs, llvm::APFixedPoint)>
        onNonTrivial = [](llvm::SmallVectorImpl<mlir::Value> &,
                          llvm::APFixedPoint) { return nullptr; }) {
  mlir::SmallVector<mlir::Value> newArgs;
  llvm::APFixedPoint constant = identity;
  int nbCst = 0;

  for (auto [arg, operand] : llvm::zip(operands, op->getOperands())) {
    /// There is a constant, so we fold it.
    if (arg) {
      fixedpt::FixedPointAttr attr = llvm::cast<fixedpt::FixedPointAttr>(arg);
      constant = calculation(constant, attr.getValue());
      nbCst++;
      continue;
    }

    newArgs.push_back(operand);
  }
  if (nbCst == 0 || (nbCst == 1 && constant != identity))
    return nullptr;

  constant =
      constant.convert(llvm::cast<FixedPtType>(op->getResult(0).getType())
                           .getFixedPointSemantics());
  if (newArgs.size() == 0)
    return fixedpt::FixedPointAttr::get(op->getContext(), constant);
  if (newArgs.size() == 1 && constant == identity)
    return newArgs[0];
  return onNonTrivial(newArgs, constant);
}

template <typename OpT>
mlir::LogicalResult canonicalizeVariadicCommutative(
    mlir::Operation *op, llvm::APFixedPoint identity,
    llvm::function_ref<llvm::APFixedPoint(llvm::APFixedPoint,
                                          llvm::APFixedPoint)>
        calculation,
    mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Attribute> constants;
  RoundingMode rounding = llvm::cast<RoundingOpInterface>(op).getRoundingMode();

  constants.assign(op->getNumOperands(), mlir::Attribute());
  for (auto [arg, attr] : llvm::zip(op->getOperands(), constants))
    mlir::matchPattern(arg, mlir::m_Constant(&attr));

  llvm::SmallVector<mlir::Value> newOperands;
  mlir::OpFoldResult result = foldVariadicCommutative(
      op, constants, identity, calculation,
      [&](llvm::SmallVectorImpl<mlir::Value> &newArgs,
          llvm::APFixedPoint constant) -> mlir::OpFoldResult {
        if (constant != identity || newArgs.size() == 0) {
          // We need to create a constant op of the resulting constant
          auto CstAttr = fitConstantBitwidth(
              fixedpt::FixedPointAttr::get(op->getContext(), constant));
          newArgs.push_back(
              rewriter.create<fixedpt::ConstantOp>(op->getLoc(), CstAttr));
        }
        assert(newArgs.size() >= 1);
        newOperands.assign(newArgs);
        return nullptr;
      });
  if (result.isNull() && newOperands.size() == 0)
    return mlir::failure();
  if (mlir::Attribute attr = result.template dyn_cast<mlir::Attribute>()) {
    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            llvm::cast<FixedPointAttr>(attr));
    return mlir::success();
  }

  if (mlir::Value val = result.template dyn_cast<mlir::Value>()) {
    assert(newOperands.size() == 0);
    newOperands.push_back(val);
  }

  if (newOperands.size() == 1)
    rewriter.replaceOpWithNewOp<fixedpt::ConvertOp>(
        op, op->getResult(0).getType(), newOperands[0], rounding);
  else
    rewriter.replaceOpWithNewOp<OpT>(op, op->getResult(0).getType(), rounding,
                                     newOperands);
  return mlir::success();
}

struct AddOpConstFolder : public mlir::RewritePattern {
  AddOpConstFolder(mlir::MLIRContext *context)
      : RewritePattern(AddOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return canonicalizeVariadicCommutative<AddOp>(
        op,
        llvm::APFixedPoint{
            0, llvm::FixedPointSemantics(1, 0, false, false, false)},
        exactAdd, rewriter);
    // if (rounding == fixedpt::RoundingMode::nearest) {
    //   llvm::APFixedPoint halfULP{
    //       1, llvm::FixedPointSemantics(
    //              1,
    //              llvm::FixedPointSemantics::Lsb{
    //                  addOp.getResult().getType().cast<FixedPtType>().getLsb()
    //                  - 1},
    //              false, false, false)};
    //   CstSumVal = exactAdd(CstSumVal, halfULP);
    //   rounding = fixedpt::RoundingMode::truncate;
    // } else
  }
};

struct MulOpConstFolder : public mlir::RewritePattern {
  MulOpConstFolder(mlir::MLIRContext *context)
      : RewritePattern(MulOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return canonicalizeVariadicCommutative<MulOp>(
        op,
        llvm::APFixedPoint{
            1, llvm::FixedPointSemantics(1, 0, false, false, false)},
        exactMul, rewriter);
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

mlir::OpFoldResult AddOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return foldVariadicCommutative(
      *this, operands,
      llvm::APFixedPoint{0,
                         llvm::FixedPointSemantics(1, 0, false, false, false)},
      exactAdd,
      [&](llvm::SmallVectorImpl<mlir::Value> &newArgs,
          llvm::APFixedPoint constant) -> mlir::OpFoldResult {
        return nullptr;
      });
}

mlir::OpFoldResult MulOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return foldVariadicCommutative(
      *this, operands,
      llvm::APFixedPoint{1,
                         llvm::FixedPointSemantics(1, 0, false, false, false)},
      exactMul,
      [&](llvm::SmallVectorImpl<mlir::Value> &newArgs,
          llvm::APFixedPoint constant) -> mlir::OpFoldResult {
        if (constant.getValue() == 0)
          return FixedPointAttr::get(getContext(), constant);
        return nullptr;
      });
}

mlir::OpFoldResult SubOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return nullptr;
}

mlir::OpFoldResult DivOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  return nullptr;
}

mlir::OpFoldResult BitcastOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  assert(operands.size() == 1);
  mlir::Type outTy = getResult().getType();
  mlir::Type inTy = getInput().getType();

  /// Same type so we remove the bitcast
  if (inTy == outTy)
    return getInput();

  /// bitcast of bitcast we can fold it to 0 or 1 bitcast
  if (auto bc = getInput().getDefiningOp<fixedpt::BitcastOp>()) {
    mlir::Type inAboveTy = bc.getInput().getType();
    if (inAboveTy == outTy)
      return bc.getInput();
    setOperand(bc.getInput());
    return bc.getResult();
  }
  return nullptr;
}

mlir::OpFoldResult ConvertOp::fold(llvm::ArrayRef<mlir::Attribute> operands)  {
  assert(operands.size() == 1);
  FixedPtType outTy = getResult().getType().cast<FixedPtType>();
  if (getInput().getType() == outTy)
    return getInput();

  if (auto cstAttr = llvm::dyn_cast_or_null<FixedPointAttr>(operands[0])) {
    llvm::APFixedPoint newVal =
        cstAttr.getValue().convert(outTy.getFixedPointSemantics());
    return FixedPointAttr::get(getContext(), std::move(newVal));
  }
  return nullptr;
}

mlir::LogicalResult ConvertOp::canonicalize(ConvertOp op, mlir::PatternRewriter& rewriter) {
  /// Simple canonicalization are implemented by fold

  mlir::Operation *opAbove = op.getInput().getDefiningOp();
  if (!opAbove)
    return mlir::failure();

  if (auto RO = llvm::dyn_cast<RoundingOpInterface>(opAbove)) {
    fixedpt::RoundingMode roundingAbove = RO.getRoundingMode();
    if (llvm::isa<AddOp, MulOp, DivOp, SubOp>(opAbove) &&
        roundingAbove == op.getRounding()) {
      RoundingOpInterface newOp = rewriter.clone(*opAbove);
      newOp->getResult(0).setType(op.getResult().getType());
      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
    if (llvm::isa<ConvertOp>(opAbove) && roundingAbove == op.getRounding()) {
      rewriter.updateRootInPlace(
          op, [&] { op.setOperand(opAbove->getOperand(0)); });
      return mlir::success();
    }
  }
  return mlir::failure();
}
