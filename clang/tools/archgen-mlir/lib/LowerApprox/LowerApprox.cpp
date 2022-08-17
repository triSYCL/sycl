//===- LowerApprox.cpp - Replace approx by its aproximation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_SOLLYA_LIB_PATH
#error "unable to find sollya"
#endif

#include "llvm/Support/DynamicLibrary.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "archgen/Approx/Approx.h"
#include "archgen/Approx/PassDetail.h"
#include "archgen/Approx/Passes.h"

#include "flopoco/FixFunctions/BasicPolyApprox.hpp"

using namespace archgen;

namespace {

/// Implementation of a generic visitor of appox expressions
template <typename RetTy> struct ApproxVisitorImpl {
  template <typename CallableTy>
  static RetTy Visit(mlir::Operation *op, CallableTy onOp) {
    if (op->getNumOperands() == 0 || llvm::isa<approx::VariableOp>(op) ||
        llvm::isa<approx::ConstantOp>(op))
      return onOp(op);
    if (op->getNumOperands() == 1)
      return onOp(op, ApproxVisitorImpl::Visit(
                          op->getOperand(0).getDefiningOp(), onOp));
    if (op->getNumOperands() == 2)
      return onOp(
          op, ApproxVisitorImpl::Visit(op->getOperand(0).getDefiningOp(), onOp),
          ApproxVisitorImpl::Visit(op->getOperand(1).getDefiningOp(), onOp));
    op->dump();
    llvm_unreachable("unhandeled op");
  }
};

template <> struct ApproxVisitorImpl<void> {
  template <typename CallableTy>
  static void Visit(mlir::Operation *rawOp, CallableTy onOp) {
    ApproxVisitorImpl<int>::Visit(rawOp, [&](auto *op, auto...) -> int {
      onOp(op);
      return 0;
    });
  }
};

/// Function to call the approx visitor
template <typename CallableTy>
auto approxTreeVisitor(mlir::Operation *rawOp, CallableTy onOp) {
  using RetTy = decltype(onOp(std::declval<mlir::Operation *>()));
  return ApproxVisitorImpl<RetTy>::Visit(rawOp, onOp);
}

struct LowerApprox {
  mlir::IRRewriter rewriter;
  mlir::MLIRContext *ctx;

  // Location of the resulting approximating expression cannot be assigned to a
  // specific part of the original expression so we consider that all of it
  // comes from evaluate
  mlir::Location loc;

  llvm::sys::DynamicLibrary sollyaLib;

  LowerApprox(mlir::MLIRContext *context)
      : rewriter(context), ctx(context), loc(rewriter.getUnknownLoc()) {}

  mlir::LogicalResult init() {
    std::string errorMsg;
    sollyaLib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        ARCHGEN_SOLLYA_LIB_PATH, &errorMsg);
    if (!sollyaLib.isValid()) {
      llvm::errs() << "error while loading sollya:" << errorMsg << "\n";
      return mlir::failure();
    }
    return mlir::success();
  }

  template <typename T> T *lookupSollyaSymbol(llvm::StringRef symbol) {
    return (T *)sollyaLib.getAddressOfSymbol(symbol.data());
  }

  template <typename... Tys>
  sollya_obj_t buildSollya(llvm::StringRef name, Tys... tys) {
    llvm::Twine fullName("sollya_lib_build_function_", name);
    return lookupSollyaSymbol<sollya_obj_t(Tys...)>(fullName.str())(tys...);
  }

  sollya_obj_t buildConstant(approx::ConstantOp op) {
    auto fixedTy = fixedpt::FixedPtType::get(
        op->getContext(), op.valueAttr().getValue().getSemantics());
    llvm::SmallVector<char> representation;
    op.valueAttr().getValue().getValue().toString(representation, 10, false);
    std::string strRep(representation.data(), representation.size());
    mpz_class constant_repr{strRep};
    mpfr_t value;
    mpfr_init2(value, fixedTy.getWidth());
    mpfr_set_z_2exp(value, constant_repr.get_mpz_t(), fixedTy.getLsb(),
                    MPFR_RNDN);
    auto ret = sollya_lib_constant(value);
    mpfr_clear(value);
    return ret;
  }

  /// Normalization inputs to [1, 0] or [1, -1]
  void normalizeInputs(approx::EvaluateOp evaluateOp) {
    approxTreeVisitor(
        evaluateOp,
        [&](mlir::Operation *rawOp, auto... operandsPack) -> mlir::Value {
          if (llvm::isa<approx::ConstantOp>(rawOp))
            return rawOp->getResult(0);
          auto op = llvm::dyn_cast<approx::VariableOp>(rawOp);
          if (!op) {
            std::array<mlir::Value, sizeof...(operandsPack)> operands = {
                operandsPack...};
            rawOp->setOperands(operands);
            assert(rawOp->getNumResults() == 1);
            return rawOp->getResult(0);
          }

          // example:
          // fixed<3, -4, u> -> fixed<0, -7, u> scaled by 2^3
          // fixed<3, -4, s> -> fixed<1, -6, s> scaled by 2^2
          rewriter.setInsertionPoint(op);
          mlir::Value variable = op->getOperand(0);

          fixedpt::FixedPtType oldType =
              op->getOperand(0).getType().cast<fixedpt::FixedPtType>();
          int newMsb = oldType.isSigned();
          int newLsb = -(oldType.getWidth() - newMsb - 1);
          fixedpt::FixedPtType newType = fixedpt::FixedPtType::get(
              ctx, newMsb, newLsb, oldType.isSigned());

          /// Already the correct type. so we bail
          if (newType == oldType)
            return op.output();

          int log2ScalingFactor = oldType.getLsb() - newType.getLsb();
          /// APFixedPoint in the current state doesn't support fixed point
          /// without a zero bit
          llvm::APFixedPoint fixedScalingFactor = [&]() -> llvm::APFixedPoint {
            if (log2ScalingFactor > 0)
              return llvm::APFixedPoint(
                  1 << log2ScalingFactor,
                  llvm::FixedPointSemantics(log2ScalingFactor + 1, 0, false, false,
                                            false));
            return llvm::APFixedPoint(
                1, llvm::FixedPointSemantics(-log2ScalingFactor + 1,
                                             -log2ScalingFactor, false, false,
                                             false));
          }();

          variable = rewriter
                         .create<fixedpt::BitcastOp>(op->getLoc(), newType,
                                                     op->getOperand(0))
                         .result();
          op.setOperand(variable);
          rewriter.setInsertionPointAfter(op);
          mlir::Value scalingFactor =
              rewriter
                  .create<approx::ConstantOp>(op.getLoc(),
                                      fixedpt::FixedPointAttr::get(
                                          ctx, std::move(fixedScalingFactor)))
                  .result();
          return rewriter
              .create<approx::GenericOp>(op->getLoc(),
                                 mlir::ValueRange{op.output(), scalingFactor},
                                 "mul")
              .output();
        });
  }

  mlir::Value findInputValue(approx::EvaluateOp evaluateOp) {
    mlir::Value inputType;
    approxTreeVisitor(evaluateOp, [&](mlir::Operation *op) {
      if (auto var = llvm::dyn_cast<approx::VariableOp>(op)) {
        assert(!inputType || (inputType == var.input()));
        inputType = var.input();
      }
    });
    return inputType;
  }

  sollya_obj_t buildSollyaTree(approx::EvaluateOp evaluateOp) {
    return approxTreeVisitor(
        evaluateOp->getOperand(0).getDefiningOp(),
        [&](auto *rawOp, auto... operands) -> sollya_obj_t {
          if (auto constant = llvm::dyn_cast<approx::ConstantOp>(rawOp))
            return buildConstant(constant);
          if (llvm::isa<approx::VariableOp>(rawOp))
            return buildSollya("free_variable");
          approx::GenericOp op = llvm::cast<approx::GenericOp>(rawOp);
          return buildSollya(op.action(), operands...);
        });
  }

  mlir::Value getConstantFromCoef(flopoco::FixConstant* constant) {
    assert(constant);
    fixedpt::FixedPtType ty = fixedpt::FixedPtType::get(
        ctx, constant->MSB, constant->LSB, constant->isSigned);
    
    llvm::APInt bits(ty.getWidth(), constant->getBitVector(), 2);
    llvm::APFixedPoint fixedVal(bits, ty.getFixedPointSemantics());

    return rewriter.create<fixedpt::ConstantOp>(
        loc, ty, fixedpt::FixedPointAttr::get(ctx, std::move(fixedVal)));
  }

  void run(approx::EvaluateOp evaluateOp) {
    loc = evaluateOp->getLoc();

    normalizeInputs(evaluateOp);

    mlir::Value output = evaluateOp.getResult();
    mlir::Value input = findInputValue(evaluateOp);
    fixedpt::FixedPtType outputType = output.getType().cast<fixedpt::FixedPtType>();
    fixedpt::FixedPtType inputType = input.getType().cast<fixedpt::FixedPtType>();

    sollya_obj_t sollyaTree = buildSollyaTree(evaluateOp);

    /// Build sollya expresion tree
    sollya_lib_printf("sollya: %b\n", sollyaTree);
    llvm::outs() << "input: width=" << inputType.getWidth()
                 << " lsb=" << inputType.getLsb() << "\n";
    llvm::outs() << "output: width=" << outputType.getWidth()
                 << " lsb=" << outputType.getLsb() << "\n";

    flopoco::BasicPolyApprox flopocoApprox(sollyaTree, std::pow(1.0, outputType.getLsb()));

    // A0 + X * ( A1 + X * (...(An-1 + X * An)))
    rewriter.setInsertionPointAfterValue(output);
    assert(flopocoApprox.getDegree() > 0);
    mlir::Value expr = getConstantFromCoef(
        flopocoApprox.getCoeff(flopocoApprox.getDegree() - 1));
    for (int i = (flopocoApprox.getDegree() - 1); i >= 0; i--) {
      mlir::Value coef = getConstantFromCoef(flopocoApprox.getCoeff(i));
      fixedpt::FixedPtType addType = inputType.getCommonAddType(
          coef.getType().cast<fixedpt::FixedPtType>());
      mlir::Value AplusX =
          rewriter.create<fixedpt::AddOp>(loc, addType, input, coef).result();
      fixedpt::FixedPtType mulType =
          addType.getCommonMulType(expr.getType().cast<fixedpt::FixedPtType>());
      if (i == 0)
        mulType = outputType;
      expr =
          rewriter.create<fixedpt::MulOp>(loc, mulType, expr, AplusX).result();
    }
    output.replaceAllUsesWith(expr);

    approxTreeVisitor(evaluateOp, [&](mlir::Operation *op) {
      op->replaceAllUsesWith(mlir::ValueRange{mlir::Value{}});
      op->erase();
    });

    expr.getParentBlock()->dump();
  }
};

struct LowerApproxPass : approx::LowerApproxPassBase<LowerApproxPass> {
  virtual void runOnOperation() override final;
};

void LowerApproxPass::runOnOperation() {
  LowerApprox state(getOperation()->getContext());
  if (mlir::failed(state.init()))
    signalPassFailure();
  getOperation().walk([&](approx::EvaluateOp op) {
    state.run(op);
  });
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::approx::createLowerApproxPass() {
  return std::make_unique<LowerApproxPass>();
}
