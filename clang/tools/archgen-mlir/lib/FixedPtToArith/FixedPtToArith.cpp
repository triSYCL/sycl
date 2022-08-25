//===- FixedPtToArith.cpp - conversion from FixedPt to Arith --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion of The FixedPt dialect to the Arith
// dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "archgen/FixedPt/PassDetail.h"
#include "archgen/FixedPt/Passes.h"

using namespace archgen;
using namespace archgen::fixedpt;
namespace arith = mlir::arith;
namespace func = mlir::func;

namespace {

//===----------------------------------------------------------------------===//
// Building blocks of the conversion
//===----------------------------------------------------------------------===//

/// Describe how FixedPt should be converted to an arith type.
struct FixedPtToArithTypeConverter : public mlir::TypeConverter {
  mlir::MLIRContext *ctx;

  /// Describe conversion for function signatures that are not handled like
  /// other types. because there might be constraints on what can be passed
  /// through function arguments so it is possible to map 1 source type to n
  /// destination types
  ///
  /// But in our case everything is simple simple 1 to 1 mapping from FixedPt
  /// types to Arith types
  ///
  /// The return value is used by FuncOpRewriting to know what it needs to
  /// rewrite the function to. TypeConverter::SignatureConversion is used by the
  /// dialect conversion framework to rewrite block operands
  mlir::FunctionType
  convertSignature(mlir::FunctionType ty,
                   TypeConverter::SignatureConversion &result) {
    /// Converted types of every arguments
    llvm::SmallVector<mlir::Type> argsTy;
    /// Converted types of every return value
    llvm::SmallVector<mlir::Type> retTy;
    argsTy.reserve(ty.getNumInputs());
    retTy.reserve(ty.getNumResults());

    int idx = 0;
    for (auto in : ty.getInputs()) {
      /// Simple 1 to 1 mapping from each type just convert them and add them to
      /// the signature description and the new FunctionType

      mlir::Type newInTy = convertType(in);
      argsTy.push_back(newInTy);
      result.addInputs(idx++, {newInTy});
    }
    for (auto out : ty.getResults()) {
      mlir::Type newOutTy = convertType(out);
      retTy.push_back(newOutTy);
    }

    return mlir::FunctionType::get(ctx, argsTy, retTy);
  }

  /// A FixedPtType is converted to its Arith equivalent
  /// for example:
  ///   !fixedpt.fixedPt<8, -7, "signed"> becomes i16
  ///   !fixedpt.fixedPt<3, -14, "unsigned"> becomes i18
  /// everything is stored in signless because it is easier. to implement this
  /// way, but obviously operations will have the correct signed or unsigned
  /// variant
  mlir::IntegerType convertFixedPt(FixedPtType ty) {
    return mlir::IntegerType::get(ctx, ty.getWidth(),
                                  mlir::IntegerType::Signless);
  }

  FixedPtToArithTypeConverter(mlir::MLIRContext *ctx) : ctx(ctx) {

    /// Add Conversions to the converter
    addConversion([&](FixedPtType ty) { return convertFixedPt(ty); });

    /// Since we generate mlir::IntegerType we need to mlir::IntegerType to be
    /// legal. so it must return it self thought convertType
    addConversion([](mlir::IntegerType ty) { return ty; });
  }
};

/// Pass class implementation
struct ConvertFixedPtToArithPass
    : ConvertFixedPtToArithPassBase<ConvertFixedPtToArithPass> {
  virtual void runOnOperation() override final;
};

/// Description of which operations are legal in input and output.
///
/// FixedPtToArith is a partial conversion, so all operation not explicitly
/// marked illegal will be left untouched. Also all operation that we generate
/// as replacement for FixedPt Ops must be explicitly legal
struct FixedPtToArithTarget : public mlir::ConversionTarget {

  /// Check if a type is legal
  static bool isLegalTypeImpl(mlir::Type ty) {
    /// Recursive case: a FunctionType is legal if all of its composing type are
    /// legal
    if (auto funcTy = ty.dyn_cast<mlir::FunctionType>())
      return isLegalType(funcTy.getInputs()) &&
             isLegalType(funcTy.getResults());

    /// Leaf case: a leaf type is legal if it is not a FixedPtType
    return !ty.isa<FixedPtType>();
  }

  /// Wrapper to easily operator on TypeRange
  static bool isLegalType(mlir::TypeRange tys) {
    return llvm::all_of(tys, isLegalTypeImpl);
  }

  FixedPtToArithTarget(mlir::MLIRContext &Ctx) : ConversionTarget(Ctx) {

    /// It needs to be converted so it must be illegal
    addIllegalDialect<FixedPtDialect>();

    /// Conversions will emit operations form ArithmeticDialect for Operations
    /// from the FixedPtDialect
    addLegalDialect<arith::ArithmeticDialect>();

    /// func::FuncOp are legal if there type is legal.
    /// We rewrite func::FuncOp only when at least one of there arguments or
    /// return is a FixedPtType
    addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp func) { return isLegalType(func.getFunctionType()); });

    /// func::ReturnOp are legal if there type is legal.
    /// We rewrite func::ReturnOp only when at least one of there arguments is
    /// a FixedPtType
    addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp ret) {
      return isLegalType(ret.operands().getTypes());
    });
  }
};

/// This is a utility class to build conversions from fixed point conversions
/// With arith types and operations
/// It also exposed some utilities convert between arith types
///
/// Note: all mlir::Value here are mlir::IntegerType
class ConversionBuilder {
  /// It will be an instance of our type converter defined above
  mlir::TypeConverter &typeConverter;

  /// Used to perform any change to the IR
  mlir::ConversionPatternRewriter &rewriter;

  /// Location of the original FixedPt op we are replacing
  mlir::Location loc;

  fixedpt::RoundingMode rounding;

public:
  ConversionBuilder(
      mlir::TypeConverter &typeConverter,
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      fixedpt::RoundingMode rounding = fixedpt::RoundingMode::zero)
      : typeConverter(typeConverter), rewriter(rewriter), loc(loc),
        rounding(rounding) {}

  /// Extend the type of v to dstTy if needed
  /// Since every mlir::IntegerType is signless the sign is given by isSigned
  mlir::Value maybeExtend(mlir::Value v, mlir::IntegerType dstTy,
                          bool isSigned) {

    /// If the destination is larger then the source we need to add an extend op
    if (v.getType().cast<mlir::IntegerType>().getWidth() < dstTy.getWidth()) {
      /// When an extention is signed or unsigned based on isSigned
      if (isSigned)
        v = rewriter
                .create<arith::ExtSIOp>(
                    loc, rewriter.getIntegerType(dstTy.getWidth()), v)
                .getOut();
      else
        v = rewriter
                .create<arith::ExtUIOp>(
                    loc, rewriter.getIntegerType(dstTy.getWidth()), v)
                .getOut();
    }
    return v;
  }

  /// Truncate the type of v to dstTy if needed
  /// truncation doesn't care about the sign. If the value fits in the result
  /// then the value in the output is the same. Otherwise it will be the N
  /// bottom bits.
  mlir::Value maybeTruncate(mlir::Value v, mlir::IntegerType dstTy) {
    if (v.getType().cast<mlir::IntegerType>().getWidth() > dstTy.getWidth())
      v = rewriter.create<arith::TruncIOp>(loc, dstTy, v).getOut();
    return v;
  }

  /// Shift v left or right depending on the requested shift
  mlir::Value relativeLeftShift(mlir::Value v, int shift, bool isSigned) {
    /// When the shift is 0 v is already we was asked for
    if (shift == 0)
      return v;

    /// Otherwise a left or right shift is need of std::abs(shift)
    /// So we create a constant with that value
    mlir::Value constShift =
        rewriter
            .create<arith::ConstantIntOp>(
                loc, std::abs(shift),
                v.getType().cast<mlir::IntegerType>().getWidth())
            .getResult();

    /// If a left shift is needed create one, sign doesn't matter
    if (shift > 0)
      return rewriter.create<arith::ShLIOp>(loc, v, constShift).getResult();

    /// Otherwise a right shift is needed. Do it signed or unsigned based on
    /// what is needed
    if (isSigned)
      return rewriter.create<arith::ShRSIOp>(loc, v, constShift).getResult();
    return rewriter.create<arith::ShRUIOp>(loc, v, constShift).getResult();
  }

  mlir::Value applyRounding(mlir::Value v, int bitsToBeRemoved, bool isSigned) {
    /// Nothing to round If we are adding bits or we round to zero
    if (bitsToBeRemoved <= 0 || rounding == fixedpt::RoundingMode::zero)
      return v;

    mlir::IntegerType ty = v.getType().cast<mlir::IntegerType>();

    assert(rounding == fixedpt::RoundingMode::nearest);
    assert(isSigned && "TODO add signed variant");

    /// v = v + (v & (1 << (bitsToBeRemoved - 1)))
    llvm::APInt mask(ty.getWidth(), 1);
    mask = mask.shl(bitsToBeRemoved - 1);

    mlir::Value maskConstant =
        rewriter
            .create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(ty, mask))
            .getResult();
    mlir::Value maskedVal =
        rewriter.create<arith::AndIOp>(loc, v, maskConstant);
    mlir::Value add =
        rewriter.create<arith::AddIOp>(loc, v, maskedVal).getResult();
    return add;
  }

  /// The top-level function to emit conversions in arith from one fixed point
  /// format to an other. This function will keep the represented fixed point
  /// value the same(if possible) between the input and output
  mlir::Value buildConversion(mlir::Value v, FixedPtType fixedSrc,
                              FixedPtType fixedDst) {
    assert(typeConverter.convertType(fixedSrc) == v.getType());

    /// The type already match, nothing to do
    if (fixedSrc == fixedDst)
      return v;

    /// Figure out the arith type of the output
    mlir::IntegerType dstTy =
        typeConverter.convertType(fixedDst).cast<mlir::IntegerType>();

    /// Extend if the output bitwidth is larger then the input bitwidth
    v = maybeExtend(v, dstTy, fixedSrc.isSigned());

    /// If the lsb was moved we need a shift
    if (fixedSrc.getLsb() != fixedDst.getLsb()) {
      v = applyRounding(v, fixedDst.getLsb() - fixedSrc.getLsb(),
                        fixedSrc.isSigned());
      v = relativeLeftShift(v, fixedSrc.getLsb() - fixedDst.getLsb(),
                            fixedSrc.isSigned());
    }

    /// Truncate if the output bitwidth is smaller then the input bitwidth
    v = maybeTruncate(v, dstTy);

    assert(v.getType() == dstTy);
    return v;
  }

  /// Simply adjust the bitwidth of v such that it becomes dstTy
  mlir::Value truncOrExtend(mlir::Value v, mlir::IntegerType dstTy,
                            bool isSigned) {
    v = maybeExtend(v, dstTy, isSigned);
    v = maybeTruncate(v, dstTy);
    assert(v.getType() == dstTy);
    return v;
  }
};

//===----------------------------------------------------------------------===//
// Descriptions of Conversion Patterns
//===----------------------------------------------------------------------===//
// For arith operations, types of lhs, rhs and result must match
// So most operator the pattern is:
// fixedpt.op(a, b) -> arith.op(convert(a), convert(b))
// convert being a conversion to the output type
//
// For each pattern:
//  - op is the original Operation
//  - adaptor is a mock operation containing the inputs the new operation should
//    have
//  - rewriter is used to performs the edit on the IR
//===----------------------------------------------------------------------===//

/// Pattern for fixedpt::AddOp to arith Ops
struct AddOpLowering : public mlir::OpConversionPattern<AddOp> {
  using base = OpConversionPattern<AddOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(AddOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.rounding());
    /// For example it converts:
    /// fixedpt.add(%a, %b) : (fixed<1,-1,u>, fixed<4,-2,s>) -> fixed<4,-1,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    /// %tmp1 = arith.extui(%a) : (i3) -> i6
    /// %tmp2 = arith.constant() {value = 1 : i7} : () -> i7
    /// %tmp3 = arith.shrsi(%b, %tmp2) : (i7, i7) -> i7
    /// %tmp4 = arith.trunci(%tmp3) : (i7) -> i6
    /// arith.addi(%tmp1, %tmp4) : (i6, i6) -> i6

    mlir::Value lhs = converter.buildConversion(
        adaptor.lhs(), op.lhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    mlir::Value rhs = converter.buildConversion(
        adaptor.rhs(), op.rhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
    return mlir::success();
  }
};

/// Pattern for fixedpt::SubOp to arith Ops
struct SubOpLowering : public mlir::OpConversionPattern<SubOp> {
  using base = OpConversionPattern<SubOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(SubOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example it converts:
    /// fixedpt.sub(%a, %b) : (fixed<4,-2,s>, fixed<3,-9,s>) -> fixed<7,-3,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    /// %tmp1 = arith.extsi(%a) : (i7) -> i11
    /// %tmp2 = arith.constant() {value = 1 : i11} : () -> i11
    /// %tmp3 = arith.shli(%tmp1, %tmp2) : (i11, i11) -> i11
    /// %tmp4 = arith.constant() {value = 6 : i13} : () -> i13
    /// %tmp5 = arith.shrsi(%b, %tmp4) : (i13, i13) -> i13
    /// %tmp6 = arith.trunci(%tmp5) : (i13) -> i11
    /// arith.subi(%tmp3, %tmp6) : (i11, i11) -> i11

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.rounding());
    mlir::Value lhs = converter.buildConversion(
        adaptor.lhs(), op.lhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    mlir::Value rhs = converter.buildConversion(
        adaptor.rhs(), op.rhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
    return mlir::success();
  }
};

/// Pattern for fixedpt::ConstantOp to arith::ConstantOp
struct ConstantOpLowering : public mlir::OpConversionPattern<ConstantOp> {
  using base = OpConversionPattern<ConstantOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ConstantOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example converts:
    ///   fixedpt.constant(){valueAttr = #fixed_point<3,1.5>} : !fixedPt<1,-1,u>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   arith.constant() {value = 3 : i3} : () -> i3

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc());
    mlir::Value v =
        rewriter
            .create<arith::ConstantOp>(
                op->getLoc(),
                mlir::IntegerAttr::get(
                    getTypeConverter()->convertType(op.result().getType()),
                    op.valueAttr().getValue().getValue()))
            .getResult();
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

/// Pattern for fixedpt::RoundOp to arith Ops
struct RoundOpLowering : public mlir::OpConversionPattern<RoundOp> {
  using base = OpConversionPattern<RoundOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult

  matchAndRewrite(RoundOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example converts:
    ///   fixedpt.round(%a) : (!fixedPt<4, -5, s>) -> !fixedPt<4, -2, s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   %tmp1 = arith.constant() {value = 3 : i10} : () -> i10
    ///   %tmp2 = arith.shrsi(%a, %tmp1) : (i10, i10) -> i10
    ///   arith.trunci(%tmp2) : (i10) -> i7

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.rounding());
    rewriter.replaceOp(op, converter.buildConversion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern for fixedpt::ExtendOp to arith Ops
struct ExtendOpLowering : public mlir::OpConversionPattern<ExtendOp> {
  using base = OpConversionPattern<ExtendOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ExtendOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example converts:
    ///   fixedpt.extend(%a) : (!fixedPt<4, -1, s>) -> !fixedPt<7, -9, s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   %tmp1 = arith.extsi(%a) : (i6) -> i17
    ///   %tmp2 = arith.constant() {value = 8 : i17} : () -> i17
    ///   arith.shli(%tmp1, %tmp2) : (i17, i17) -> i17

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc());
    rewriter.replaceOp(op, converter.buildConversion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern for fixedpt::BitcastOp to nothing. the operand after type rewriting
/// should always be the same type as the expected output
struct BitcastOpLowering : public mlir::OpConversionPattern<BitcastOp> {
  using base = OpConversionPattern<BitcastOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(BitcastOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// The type should already be correct in the adaptor
    assert(adaptor.input().getType() ==
           typeConverter->convertType(op.result().getType()));
    rewriter.replaceOp(op, adaptor.getOperands());
    return mlir::success();
  }
};

/// Pattern for fixedpt::TruncOp to arith::TruncIOp
struct TruncOpLowering : public mlir::OpConversionPattern<TruncOp> {
  using base = OpConversionPattern<TruncOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(TruncOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    /// For example converts:
    ///   fixedpt.trunc(%a) : (!fixedPt<7, -9, u>) -> !fixedPt<3, -9, s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   arith.trunci(%a) : (i17) -> i13

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc());
    rewriter.replaceOp(op, converter.buildConversion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern for fixedpt::TruncOp to arith Ops
struct MulOpLowering : public mlir::OpConversionPattern<MulOp> {
  using base = OpConversionPattern<MulOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(MulOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// FIXME: The mul is performed on a bitwidth that is much larger then it
    /// could be

    /// For example converts:
    ///   fixedpt.mul(%a, %b):(!fixed<4,-2,s>,!fixed<3,-9,s>) -> !fixed<7,-5,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   %tmp1 = arith.extsi(%4) : (i7) -> i19
    ///   %tmp2 = arith.extsi(%18) : (i13) -> i19
    ///   %tmp3 = arith.muli(%tmp1, %tmp2) : (i19, i19) -> i19
    ///   %tmp4 = arith.constant() {value = 6 : i19} : () -> i19
    ///   %tmp5 = arith.shrsi(%tmp3, %tmp4) : (i19, i19) -> i19
    ///   arith.trunci(%tmp5) : (i19) -> i13

    FixedPtType lhsTy = op.lhs().getType().cast<FixedPtType>();
    FixedPtType rhsTy = op.rhs().getType().cast<FixedPtType>();

    /// Figure out the fixed point type resulting from a product of lhs and rhs
    FixedPtType internalTy = lhsTy.getCommonMulType(rhsTy);
    /// And its arith lowering
    mlir::IntegerType internalIntTy =
        typeConverter->convertType(internalTy).cast<mlir::IntegerType>();

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.rounding());
    /// reinterpret extend the withd of lhs and rhs to fit the internalIntTy
    mlir::Value lhs =
        converter.maybeExtend(adaptor.lhs(), internalIntTy, lhsTy.isSigned());
    mlir::Value rhs =
        converter.maybeExtend(adaptor.rhs(), internalIntTy, rhsTy.isSigned());
    /// Multiply with it
    auto mulOp = rewriter.create<arith::MulIOp>(op.getLoc(), lhs, rhs);

    /// Convert result to the requested size
    rewriter.replaceOp(op, converter.buildConversion(
                               mulOp.getResult(), internalTy,
                               op.getResult().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern for fixedpt::TruncOp to arith Ops
struct DivOpLowering : public mlir::OpConversionPattern<DivOp> {
  using base = OpConversionPattern<DivOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(DivOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example converts:
    ///   fixedpt.div(%a, %b):(!fixed<7,-5,s>, !fixed<7,-3,s>) -> !fixed<7,-9,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///    %tmp1 = arith.extsi(%a) : (i13) -> i20
    ///    %tmp2 = arith.constant() {value = 7 : i20} : () -> i20
    ///    %tmp3 = arith.shli(%tmp1, %tmp2) : (i20, i20) -> i20
    ///    %tmp4 = arith.extsi(%b) : (i11) -> i20
    ///    %tmp4 = arith.divsi(%tmp3, %tmp4) : (i20, i20) -> i20
    ///    arith.trunci(%tmp4) : (i20) -> i17

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.rounding());
    FixedPtType lhsTy = op.lhs().getType().cast<FixedPtType>();
    FixedPtType rhsTy = op.rhs().getType().cast<FixedPtType>();
    FixedPtType outTy = op.result().getType().cast<FixedPtType>();
    bool isSigned = lhsTy.isSigned() || rhsTy.isSigned();
    mlir::Value lhs;
    mlir::Value rhs;

    /// precision If we divide without shifting inputs
    int currentPrec = lhsTy.getLsb() - rhsTy.getLsb();

    /// how much lsb do we need to add to lhs to have an output with the right
    /// precision
    int lsbOffset = outTy.getLsb() - currentPrec;

    /// If we need to increase precision
    if (lsbOffset <= 0) {
      /// width of the division operation
      int divWidth = std::max(lhsTy.getWidth() - lsbOffset, rhsTy.getWidth());
      int newLsb = lhsTy.getLsb() + lsbOffset;

      /// Convert lhs and rhs to their new formats (or not)
      lhs = converter.buildConversion(
          adaptor.lhs(), lhsTy,
          FixedPtType::get(rewriter.getContext(), divWidth + newLsb - 1, newLsb,
                           lhsTy.isSigned()));
      rhs = converter.maybeExtend(
          adaptor.rhs(), rewriter.getIntegerType(divWidth), rhsTy.isSigned());
    } else {
      /// In this case we need to artificially reduce the precision of the
      /// division to fit in the output without calculating useless bits
      op->dump();
      llvm_unreachable("unimplemented");
    }

    mlir::Value divResult;
    /// Emit the division
    if (isSigned)
      divResult =
          rewriter.create<arith::DivSIOp>(op.getLoc(), lhs, rhs).getResult();
    else
      divResult =
          rewriter.create<arith::DivUIOp>(op.getLoc(), lhs, rhs).getResult();

    /// Remove the now useless high bits
    rewriter.replaceOp(
        op, converter.maybeTruncate(
                divResult,
                typeConverter->convertType(outTy).cast<mlir::IntegerType>()));
    return mlir::success();
  }
};

/// Pattern for fixedpt::TruncOp to arith::TruncIOp
struct ConvertOpLowering : public mlir::OpConversionPattern<ConvertOp> {
  using base = OpConversionPattern<ConvertOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ConvertOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    /// For example converts:
    ///   fixedpt.convert(%a) : (!fixedPt<7, -9, u>) -> !fixedPt<3, -9, s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   arith.trunci(%a) : (i17) -> i13

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.rounding());
    rewriter.replaceOp(op, converter.buildConversion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern to rewrite the type and blocks of a func::FuncOp
struct FuncOpRewriting : public mlir::OpConversionPattern<func::FuncOp> {
  using base = OpConversionPattern<func::FuncOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(func::FuncOp oldFunc, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// Figure out the new type and arguments rewrite rules
    mlir::TypeConverter::SignatureConversion result(
        oldFunc.getFunctionType().getNumInputs());
    mlir::FunctionType newFuncTy =
        getTypeConverter<FixedPtToArithTypeConverter>()->convertSignature(
            oldFunc.getFunctionType(), result);

    /// Create the new function with the new type and same old symbol name
    auto newFunc = rewriter.create<func::FuncOp>(
        oldFunc.getLoc(), oldFunc.getSymName(), newFuncTy);
    /// Move regions from the old function operations to the new
    rewriter.inlineRegionBefore(oldFunc.getRegion(), newFunc.getRegion(),
                                newFunc.getRegion().end());

    /// Rewrite regions block arguments to have new types
    if (mlir::failed(rewriter.convertRegionTypes(&newFunc.getBody(),
                                                 *typeConverter, &result)))
      return mlir::failure();

    /// Replace uses of oldFunc and notify rewriter we are done
    rewriter.replaceOp(oldFunc, newFunc->getResults());

    return mlir::success();
  }
};

/// Pattern to rewrite the type of a func::ReturnOp
struct ReturnOpRewriting : public mlir::OpConversionPattern<func::ReturnOp> {
  using base = OpConversionPattern<func::ReturnOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(func::ReturnOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// The types in the adaptor are already correct so we just create a new
    /// ReturnOp with them
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.operands());
    return mlir::success();
  }
};

/// Fill the RewritePatternSet with our rewrite patterns
void populateFixedPtToArithConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::TypeConverter &converter) {
  // clang-format off
  patterns.add<AddOpLowering,
               SubOpLowering,
               ConstantOpLowering,
               RoundOpLowering,
               ExtendOpLowering,
               BitcastOpLowering,
               TruncOpLowering,
               MulOpLowering,
               DivOpLowering,
               ConvertOpLowering,
               FuncOpRewriting,
               ReturnOpRewriting
               >(converter, patterns.getContext());
  // clang-format on
}

void ConvertFixedPtToArithPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  FixedPtToArithTarget target(getContext());
  FixedPtToArithTypeConverter typeConverter(&getContext());

  populateFixedPtToArithConversionPatterns(patterns, typeConverter);

  /// Apply our rewrite patterns and thats it
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::fixedpt::createConvertFixedPtToArithPass() {
  return std::make_unique<ConvertFixedPtToArithPass>();
}
