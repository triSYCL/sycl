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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"

#include "archgen/FixedPt/FixedPt.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

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
  /// The common type with null is the non-null type;
  if (!*this || !other) {
    FixedPtType res = (!other) ? *this : other;
    assert(res);
    return res;
  }
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
  /// The common type with null is the non-null type;
  if (!*this || !other) {
    FixedPtType res = (!other) ? *this : other;
    assert(res);
    return res;
  }
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

namespace {

void printVariadicOp(mlir::Operation *op, mlir::OpAsmPrinter &printer) {
  bool isFirst = true;
  printer << " ";
  for (mlir::Value v : op->getOperands()) {
    printer << (isFirst ? (isFirst = false, "") : ", ") << v << " : ";
    printer.printStrippedAttrOrType(v.getType().cast<FixedPtType>());
  }
  printer << " to "
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
  if (parser.parseKeyword("to") || parser.getCurrentLocation(&roundKWLoc) ||
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

  llvm::APInt intPart = rawInt.zextOrTrunc(ty.cast<FixedPtType>().getWidth());
  if (rawInt != intPart.zextOrTrunc(rawInt.getBitWidth()))
    return parser.emitError(intLoc, "integer doesn't fit in format");
  llvm::APFixedPoint value(intPart,
                           ty.cast<FixedPtType>().getFixedPointSemantics());
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
    size_t NbCst{0};
    for (auto arg : addOp.args()) {
      if (mlir::matchPattern(arg, mlir::m_Constant(&constVal))) {
        auto CstValue = constVal.getValue();
        if (!CstValue.getValue().isZero()) {
          if (!CstSumVal.getValue().isZero()) {
            CstSumVal = exactAdd(CstSumVal, CstValue);
          } else {
            // To avoid useless bit extension due to the arbitrary choice of
            // zero semantics
            CstSumVal = CstValue;
          }
          NbCst++;
        }
      } else {
        NewArgs.push_back(arg);
      }
    }
    if (NbCst == 0 || (NbCst == 1 && !CstSumVal.getValue().isZero())) {
      return mlir::failure();
    }

    if (!CstSumVal.getValue().isZero() || NewArgs.size() == 0) {
      // We need to create a constant op of the resulting constant
      auto CstType = FixedPtType::get(getContext(), CstSumVal.getSemantics());
      auto CstAttr = fixedpt::FixedPointAttr::get(getContext(), CstSumVal);
      auto CstValue =
          rewriter.create<fixedpt::ConstantOp>(op->getLoc(), CstType, CstAttr)
              .result();
      NewArgs.push_back(CstValue);
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

} // namespace

void MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<MergeVariadicCommutativeOp<MulOp>>(context);
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
  auto ConstVal = op.valueAttr();
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

  if (NewLsb == InitLsb && NewMsb == InitMsb && NewSign == InitIsSigned) {
    // No changes
    return mlir::failure();
  }

  assert(NewMsb >= NewLsb);

  ConstAPInt.setIsSigned(NewSign);

  if (!ConstAPInt.isZero())
    ConstAPInt.ashrInPlace(CTZ);

  unsigned int NewWidth{static_cast<unsigned int>(NewMsb - NewLsb) + 1};

  auto NewConstRepr = ConstAPInt.trunc(NewWidth);

  llvm::FixedPointSemantics NewFormat{
      NewWidth, llvm::FixedPointSemantics::Lsb{NewLsb}, NewSign, false, false};
  llvm::APFixedPoint NewCstFix{NewConstRepr, NewFormat};
  auto NewCstFixType = FixedPtType::get(op.getContext(), NewFormat);

  auto ReplAttr = fixedpt::FixedPointAttr::get(op->getContext(), std::move(NewCstFix));
  rewriter.replaceOpWithNewOp<fixedpt::ConstantOp>(op, NewCstFixType, ReplAttr);
  return mlir::success();
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
  FixedPtType outTy = op.result().getType().cast<FixedPtType>();
  if (op.input().getType() == outTy) {
    rewriter.replaceOp(op, op.input());
    return mlir::success();
  }
  mlir::Operation *opAbove = op.input().getDefiningOp();
  if (!opAbove)
    return mlir::failure();
  if (llvm::isa<AddOp, MulOp, DivOp, SubOp>(opAbove)) {
    RoundingOpInterface newOp = rewriter.clone(*opAbove);
    newOp->getResult(0).setType(op.result().getType());
    newOp.setRoundingMode(getCommonRoundingMod(
        op.rounding(),
        op.input().getDefiningOp<RoundingOpInterface>().getRoundingMode()));
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
  if (llvm::isa<TruncOp, RoundOp, ConvertOp, ExtendOp>(opAbove)) {
    rewriter.updateRootInPlace(op, [&] {
      op.setOperand(opAbove->getOperand(0));
      RoundingMode rounding = op.getRoundingMode();
      if (auto RO = llvm::dyn_cast<RoundingOpInterface>(opAbove))
        rounding = getCommonRoundingMod(rounding, RO.getRoundingMode());
      op.setRoundingMode(rounding);
    });
    return mlir::success();
  }
  if (ConstantOp cstOp = llvm::dyn_cast<fixedpt::ConstantOp>(opAbove)) {
    llvm::APFixedPoint newVal =
        cstOp.valueAttr().getValue().convert(outTy.getFixedPointSemantics());
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, outTy,
        FixedPointAttr::get(rewriter.getContext(), std::move(newVal)));
    return mlir::success();
  }
  return mlir::failure();
}
