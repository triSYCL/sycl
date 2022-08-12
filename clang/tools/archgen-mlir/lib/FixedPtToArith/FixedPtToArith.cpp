//===- FixedPtToArith.cpp - conversion from FixedPt to Arith --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

struct FixedPtToAirthTypeConverter : public mlir::TypeConverter {
  mlir::MLIRContext *ctx;

  mlir::FunctionType
  convertSignature(mlir::FunctionType ty,
                   TypeConverter::SignatureConversion &result) {
    llvm::SmallVector<mlir::Type> argsTy;
    llvm::SmallVector<mlir::Type> retTy;
    argsTy.reserve(ty.getNumInputs());
    retTy.reserve(ty.getNumResults());
    int idx = 0;
    for (auto in : ty.getInputs()) {
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

  mlir::IntegerType convertFixedPt(FixedPtType ty) {
    return mlir::IntegerType::get(ctx, ty.getWidth(),
                                  mlir::IntegerType::Signless);
  }

  FixedPtToAirthTypeConverter(mlir::MLIRContext *ctx) : ctx(ctx) {
    addConversion([&](FixedPtType ty) { return convertFixedPt(ty); });
    addConversion([](mlir::IntegerType ty) { return ty; });
  }
};

struct ConvertFixedPtToArithPass
    : ConvertFixedPtToArithPassBase<ConvertFixedPtToArithPass> {
  virtual void runOnOperation() override final;
};

struct FixedPtToArithTarget : public mlir::ConversionTarget {
  static bool isLegalType(mlir::TypeRange tys) {
    return llvm::all_of(tys, isLegalTypeImpl);
  }
  static bool isLegalTypeImpl(mlir::Type ty) {
    if (auto funcTy = ty.dyn_cast<mlir::FunctionType>())
      return isLegalType(funcTy.getInputs()) &&
             isLegalType(funcTy.getResults());
    return !ty.isa<FixedPtType>();
  }
  FixedPtToArithTarget(mlir::MLIRContext &Ctx) : ConversionTarget(Ctx) {
    addIllegalDialect<FixedPtDialect>();
    addLegalDialect<arith::ArithmeticDialect, mlir::BuiltinDialect>();
    addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      return isLegalType(func.getFunctionType());
    });
    addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp ret) {
      return isLegalType(ret.operands().getTypes());
    });
  }
};

class ConvertionBuilder {
  mlir::TypeConverter &typeConverter;
  mlir::ConversionPatternRewriter &rewriter;
  mlir::Location loc;

public:
  ConvertionBuilder(mlir::TypeConverter &typeConverter,
                    mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc)
      : typeConverter(typeConverter), rewriter(rewriter), loc(loc) {}

  mlir::Value maybeExtend(mlir::Value v, mlir::IntegerType dstTy,
                          bool isSigned) {
    if (v.getType().cast<mlir::IntegerType>().getWidth() < dstTy.getWidth()) {
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

  mlir::Value maybeTruncate(mlir::Value v, mlir::IntegerType dstTy) {
    if (v.getType().cast<mlir::IntegerType>().getWidth() > dstTy.getWidth())
      v = rewriter.create<arith::TruncIOp>(loc, dstTy, v).getOut();
    return v;
  }

  mlir::Value relativeLeftShift(mlir::Value v, int shift) {
    if (shift == 0)
      return v;
    mlir::Value constShift =
        rewriter
            .create<arith::ConstantIntOp>(
                loc, std::abs(shift),
                v.getType().cast<mlir::IntegerType>().getWidth())
            .getResult();
    if (shift > 0)
      return rewriter.create<arith::ShLIOp>(loc, v, constShift).getResult();
    if (v.getType().cast<mlir::IntegerType>().isSigned())
      return rewriter.create<arith::ShRSIOp>(loc, v, constShift).getResult();
    return rewriter.create<arith::ShRUIOp>(loc, v, constShift).getResult();
  }

  mlir::Value buildConvertion(mlir::Value v, FixedPtType fixedSrc,
                              FixedPtType fixedDst) {
    assert(typeConverter.convertType(fixedSrc) == v.getType());
    if (fixedSrc == fixedDst)
      return v;
    mlir::IntegerType dstTy =
        typeConverter.convertType(fixedDst).cast<mlir::IntegerType>();
    v = maybeExtend(v, dstTy, fixedSrc.isSigned());
    if (fixedSrc.getLsb() != fixedDst.getLsb())
      v = relativeLeftShift(v, fixedSrc.getLsb() - fixedDst.getLsb());
    v = maybeTruncate(v, dstTy);
    assert(v.getType() == dstTy);
    return v;
  }
  mlir::Value truncOrExtend(mlir::Value v, mlir::IntegerType dstTy, bool isSigned) {
    v = maybeExtend(v, dstTy, isSigned);
    v = maybeTruncate(v, dstTy);
    assert(v.getType() == dstTy);
    return v;
  }
};

struct AddOpLowering : public mlir::OpConversionPattern<AddOp> {
  using base = OpConversionPattern<AddOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(AddOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    mlir::Value lhs = converter.buildConvertion(
        adaptor.lhs(), op.lhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    mlir::Value rhs = converter.buildConvertion(
        adaptor.rhs(), op.rhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
    return mlir::success();
  }
};

struct SubOpLowering : public mlir::OpConversionPattern<SubOp> {
  using base = OpConversionPattern<SubOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(SubOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    mlir::Value lhs = converter.buildConvertion(
        adaptor.lhs(), op.lhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    mlir::Value rhs = converter.buildConvertion(
        adaptor.rhs(), op.rhs().getType().cast<FixedPtType>(),
        op.result().getType().cast<FixedPtType>());
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
    return mlir::success();
  }
};

struct ConstantOpLowering : public mlir::OpConversionPattern<ConstantOp> {
  using base = OpConversionPattern<ConstantOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ConstantOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
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

struct RoundOpLowering : public mlir::OpConversionPattern<RoundOp> {
  using base = OpConversionPattern<RoundOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(RoundOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    rewriter.replaceOp(op, converter.buildConvertion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

struct ExtendOpLowering : public mlir::OpConversionPattern<ExtendOp> {
  using base = OpConversionPattern<ExtendOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ExtendOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    rewriter.replaceOp(op, converter.buildConvertion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

struct BitcastOpLowering : public mlir::OpConversionPattern<BitcastOp> {
  using base = OpConversionPattern<BitcastOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(BitcastOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type dstTy = op.result().getType().isa<FixedPtType>()
                           ? typeConverter->convertType(op.result().getType())
                           : op.result().getType();
    mlir::ValueRange replacement = adaptor.getOperands();
    if (adaptor.input().getType() != dstTy)
      replacement =
          rewriter
              .create<arith::BitcastOp>(op->getLoc(), dstTy, adaptor.input())
              ->getResults();
    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }
};

struct TruncOpLowering : public mlir::OpConversionPattern<TruncOp> {
  using base = OpConversionPattern<TruncOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(TruncOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    rewriter.replaceOp(op, converter.buildConvertion(
                               adaptor.input(),
                               op.input().getType().cast<FixedPtType>(),
                               op.result().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

struct MulOpLowering : public mlir::OpConversionPattern<MulOp> {
  using base = OpConversionPattern<MulOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(MulOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    FixedPtType lhsTy = op.lhs().getType().cast<FixedPtType>();
    FixedPtType rhsTy = op.rhs().getType().cast<FixedPtType>();
    FixedPtType internalTy = FixedPtType::get(
        rewriter.getContext(), lhsTy.getMsb() + rhsTy.getMsb(),
        lhsTy.getLsb() + rhsTy.getLsb(), lhsTy.isSigned() || rhsTy.isSigned());
    mlir::IntegerType internalIntTy =
        typeConverter->convertType(internalTy).cast<mlir::IntegerType>();

    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    mlir::Value lhs =
        converter.maybeExtend(adaptor.lhs(), internalIntTy, lhsTy.isSigned());
    mlir::Value rhs =
        converter.maybeExtend(adaptor.rhs(), internalIntTy, rhsTy.isSigned());
    auto mulOp = rewriter.create<arith::MulIOp>(op.getLoc(), lhs, rhs);

    rewriter.replaceOp(op, converter.buildConvertion(
                               mulOp.getResult(), internalTy,
                               op.getResult().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

struct DivOpLowering : public mlir::OpConversionPattern<DivOp> {
  using base = OpConversionPattern<DivOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(DivOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConvertionBuilder converter(*typeConverter, rewriter, op->getLoc());
    FixedPtType lhsTy = op.lhs().getType().cast<FixedPtType>();
    FixedPtType rhsTy = op.rhs().getType().cast<FixedPtType>();
    FixedPtType outTy = op.result().getType().cast<FixedPtType>();
    bool isSigned = lhsTy.isSigned() || rhsTy.isSigned();
    mlir::Value lhs;
    mlir::Value rhs;

    int currentPrec = lhsTy.getLsb() - rhsTy.getLsb();
    int lsbOffset = outTy.getLsb() - currentPrec;
    if (lsbOffset <= 0) {
      int divWidth = std::max(lhsTy.getWidth() - lsbOffset, rhsTy.getWidth());
      int newLsb = lhsTy.getLsb() + lsbOffset;
      lhs = converter.buildConvertion(
          adaptor.lhs(), lhsTy,
          FixedPtType::get(rewriter.getContext(), divWidth + newLsb - 1, newLsb,
                           lhsTy.isSigned()));
      rhs = converter.maybeExtend(
          adaptor.rhs(), rewriter.getIntegerType(divWidth), rhsTy.isSigned());
    } else {
      /// In this case we need to artificially reduce the precision of the
      /// division to fit in the output
      op->dump();
      llvm_unreachable("unimplemented");
    }

    mlir::Operation *divOp;
    if (isSigned)
      divOp = rewriter.create<arith::DivSIOp>(op.getLoc(), lhs, rhs);
    else
      divOp = rewriter.create<arith::DivUIOp>(op.getLoc(), lhs, rhs);

    rewriter.replaceOp(
        op, converter.maybeTruncate(
                divOp->getResult(0),
                typeConverter->convertType(outTy).cast<mlir::IntegerType>()));
    return mlir::success();
  }
};

struct FuncOpRewriting : public mlir::OpConversionPattern<func::FuncOp> {
  using base = OpConversionPattern<func::FuncOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(func::FuncOp oldFunc, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter::SignatureConversion result(
        oldFunc.getFunctionType().getNumInputs());
    mlir::FunctionType newFuncTy =
        getTypeConverter<FixedPtToAirthTypeConverter>()->convertSignature(
            oldFunc.getFunctionType(), result);

    auto newFunc = rewriter.create<func::FuncOp>(
        oldFunc.getLoc(), oldFunc.getSymName(), newFuncTy);
    rewriter.inlineRegionBefore(oldFunc.getRegion(), newFunc.getRegion(),
                                newFunc.getRegion().end());

    if (mlir::failed(rewriter.convertRegionTypes(&newFunc.getBody(),
                                                 *typeConverter, &result)))
      return mlir::failure();

    rewriter.replaceOp(oldFunc, newFunc->getResults());

    return mlir::success();
  }
};

struct ReturnOpRewriting : public mlir::OpConversionPattern<func::ReturnOp> {
  using base = OpConversionPattern<func::ReturnOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(func::ReturnOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.operands());
    return mlir::success();
  }
};

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
               FuncOpRewriting,
               ReturnOpRewriting
               >(converter, patterns.getContext());
  // clang-format on
}

void ConvertFixedPtToArithPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  FixedPtToArithTarget target(getContext());
  FixedPtToAirthTypeConverter typeConverter(&getContext());

  populateFixedPtToArithConversionPatterns(patterns, typeConverter);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::fixedpt::createConvertFixedPtToArithPass() {
  return std::make_unique<ConvertFixedPtToArithPass>();
}
