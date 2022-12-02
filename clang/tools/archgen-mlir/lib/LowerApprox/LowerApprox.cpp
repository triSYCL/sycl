//===- LowerApprox.cpp - Replace approx by its approximation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_SOLLYA_LIB_PATH
#error "unable to find sollya"
#endif

#include <cmath>
#include <vector>

#include "flopoco/FixConstant.hpp"
#include "flopoco/FixFunctions/FixFunction.hpp"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/DynamicLibrary.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "archgen/Approx/Approx.h"
#include "archgen/Approx/PassDetail.h"
#include "archgen/Approx/Passes.h"
#include "archgen/FixedPt/FixedPt.h"

#include "flopoco/FixFunctions/BasicPolyApprox.hpp"
#include "flopoco/FixFunctions/FixHorner.hpp"
#include "flopoco/Targets/VirtexUltrascalePlus.hpp"
#include "flopoco/report.hpp"

using namespace archgen;
namespace LLVM = mlir::LLVM;
namespace arith = mlir::arith;

namespace {

/// Implementation of a generic visitor of approx expressions
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

/// Pass implementation
struct LowerApprox {
  mlir::IRRewriter rewriter;
  mlir::MLIRContext *ctx;

  // Location of the resulting approximating expression cannot be assigned to a
  // specific part of the original expression so we consider that all of it
  // comes from evaluate
  mlir::Location loc;

  /// To do calls by function name into sollya we load it as a dynamic library
  llvm::sys::DynamicLibrary sollyaLib;

  LowerApprox(mlir::MLIRContext *context)
      : rewriter(context), ctx(context), loc(rewriter.getUnknownLoc()) {}

  /// Load Sollya
  mlir::LogicalResult init() {
    std::string errorMsg;

    /// cmake will kindly give us the path of the sollya library it selected
    sollyaLib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        ARCHGEN_SOLLYA_LIB_PATH, &errorMsg);
    if (!sollyaLib.isValid()) {
      llvm::errs() << "error while loading sollya:" << errorMsg << "\n";
      return mlir::failure();
    }
    return mlir::success();
  }

  /// Lookup a symbol of a specific type in sollya
  template <typename T> T *lookupSollyaSymbol(llvm::StringRef symbol) {
    return (T *)sollyaLib.getAddressOfSymbol(symbol.data());
  }

  /// Build a sollya expression
  template <typename... Tys>
  sollya_obj_t buildSollya(llvm::StringRef name, Tys... tys) {
    llvm::Twine fullName("sollya_lib_build_function_", name);
    return lookupSollyaSymbol<sollya_obj_t(Tys...)>(fullName.str())(tys...);
  }

  /// Build a sollya constant
  sollya_obj_t buildConstant(approx::ConstantOp op) {
    auto fixedTy = fixedpt::FixedPtType::get(
        op->getContext(), op.getValueAttr().getValue().getSemantics());
    llvm::SmallVector<char> representation;
    op.getValueAttr().getValue().getValue().toString(representation, 10, false);
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

  void normalizeInputs(approx::EvaluateOp evaluateOp) {
    approxTreeVisitor(
        evaluateOp,
        [&](mlir::Operation *rawOp, auto... operandsPack) -> mlir::Value {
          /// Handle recursive rewrites.
          if (llvm::isa<approx::ConstantOp>(rawOp))
            return rawOp->getResult(0);

          auto op = llvm::dyn_cast<approx::VariableOp>(rawOp);
          if (!op) {
            /// If this is not a variable set the operands from the operand pack
            /// Such that we can return a new value and this will update the
            /// users of it
            std::array<mlir::Value, sizeof...(operandsPack)> operands = {
                operandsPack...};
            rawOp->setOperands(operands);
            assert(rawOp->getNumResults() == 1);
            return rawOp->getResult(0); // all out ops only have one result
          }

          /// Handle normalization
          /// example:
          /// fixed<3, -4, u> -> fixed<-1, -8, u> scaled by 2^4
          /// fixed<3, -4, s> -> fixed<0, -7, s> scaled by 2^3

          /// We are going to start editing so set where we want to write
          rewriter.setInsertionPoint(op);

          /// get the fixe point input to the expression
          mlir::Value variable = op->getOperand(0);
          fixedpt::FixedPtType oldType =
              op->getOperand(0).getType().cast<fixedpt::FixedPtType>();

          /// figure out the new type we want such that it is either [1, 0] or
          /// [1, -1]
          int newMsb = (int)oldType.isSigned() - 1;
          int newLsb = -(oldType.getWidth() - newMsb - 1);
          fixedpt::FixedPtType newType = fixedpt::FixedPtType::get(
              ctx, newMsb, newLsb, oldType.isSigned());

          /// Already the correct type. so we bail
          if (newType == oldType)
            return op.getOutput();

          /// figure out the scaling constant
          int log2ScalingFactor = oldType.getLsb() - newType.getLsb();
          llvm::APFixedPoint fixedScalingFactor = [&]() -> llvm::APFixedPoint {
            if (log2ScalingFactor > 0)
              return llvm::APFixedPoint(
                  1 << log2ScalingFactor,
                  llvm::FixedPointSemantics(log2ScalingFactor + 1, 0, false,
                                            false, false));
            return llvm::APFixedPoint(
                1, llvm::FixedPointSemantics(-log2ScalingFactor + 1,
                                             -log2ScalingFactor, false, false,
                                             false));
          }();

          /// bitcast to convert from the input format to a normalized format
          /// for free
          variable = rewriter
                         .create<fixedpt::BitcastOp>(op->getLoc(), newType,
                                                     op->getOperand(0))
                         .getResult();

          /// This is the new input to the expression
          op.setOperand(variable);

          /// build the scaling constant
          rewriter.setInsertionPointAfter(op);
          mlir::Value scalingFactor =
              rewriter
                  .create<approx::ConstantOp>(
                      op.getLoc(), fixedpt::FixedPointAttr::get(
                                       ctx, std::move(fixedScalingFactor)))
                  .getResult();

          /// build the multiply
          return rewriter
              .create<approx::GenericOp>(
                  op->getLoc(), mlir::ValueRange{op.getOutput(), scalingFactor},
                  "mul")
              .getOutput();
        });
  }

  mlir::Value findInputValue(approx::EvaluateOp evaluateOp) {
    mlir::Value inputVal;
    approxTreeVisitor(evaluateOp, [&](mlir::Operation *op) {
      if (auto var = llvm::dyn_cast<approx::VariableOp>(op)) {
        assert(!inputVal || (inputVal == var.getInput()));
        inputVal = var.getInput();
      }
    });
    return inputVal;
  }

  sollya_obj_t buildSollyaTree(approx::EvaluateOp evaluateOp) {
    return approxTreeVisitor(
        // skip the approx::EvaluateOp since we dont care about it
        evaluateOp->getOperand(0).getDefiningOp(),
        [&](auto *rawOp, auto... operands) -> sollya_obj_t {
          if (auto constant = llvm::dyn_cast<approx::ConstantOp>(rawOp))
            return buildConstant(constant);
          if (llvm::isa<approx::VariableOp>(rawOp))
            return buildSollya("free_variable");
          approx::GenericOp op = llvm::cast<approx::GenericOp>(rawOp);
          return buildSollya(op.getAction(), operands...);
        });
  }

  std::pair<llvm::APFixedPoint, fixedpt::FixedPtType>
  getCoefAsApFixed(flopoco::FixConstant *constant) {
    assert(constant);
    fixedpt::FixedPtType ty = fixedpt::FixedPtType::get(
        ctx, constant->MSB, constant->LSB, constant->isSigned);

    std::string quotedBitsStr = constant->getBitVector();
    llvm::StringRef bitStr = quotedBitsStr;
    bitStr = bitStr.drop_front().drop_back();
    llvm::APInt bits(ty.getWidth(), bitStr, 2);
    llvm::APFixedPoint fixedVal(bits, ty.getFixedPointSemantics());
    return {fixedVal, ty};
  }

  /// build a constant from a flopoco::FixConstant*
  mlir::Value getConstantFromCoef(flopoco::FixConstant *constant) {
    std::pair<llvm::APFixedPoint, fixedpt::FixedPtType> fixedCst =
        getCoefAsApFixed(constant);
    return rewriter.create<fixedpt::ConstantOp>(
        loc, fixedCst.second,
        fixedpt::FixedPointAttr::get(ctx, std::move(fixedCst.first)));
  }

  void printPolynom(llvm::raw_ostream &os,
                    llvm::ArrayRef<flopoco::FixConstant *> coefs) {
    std::string expr = getCoefAsApFixed(coefs.back()).first.toString();
    for (auto *coef : llvm::reverse(coefs.drop_back())) {
      std::string cst = getCoefAsApFixed(coef).first.toString();
      expr = cst + " + x * " + "(" + expr + ")";
    }
    os << "polynom:\ndef f(x):\n\treturn " << expr << "\n";
  }

  static void printFormat(llvm::StringRef str, fixedpt::FixedPtType ty) {
    llvm::outs() << str << ": width=" << ty.getWidth() << " lsb=" << ty.getLsb()
                 << (ty.isSigned() ? " signed" : " unsigned") << "\n";
  }

  LLVM::LLVMFuncOp printer;
  void emitPrintCall(mlir::Value v, int id) {
    mlir::ModuleOp module =
        v.getParentBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (!printer) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      printer = rewriter.create<LLVM::LLVMFuncOp>(
          loc, "debug_printer",
          LLVM::LLVMFunctionType::get(
              LLVM::LLVMVoidType::get(rewriter.getContext()),
              {rewriter.getI32Type(), rewriter.getI32Type(),
               rewriter.getI32Type()}));
    }

    int bitwidth = v.getType().cast<fixedpt::FixedPtType>().getWidth();
    mlir::Value bitwidthConst =
        rewriter
            .create<arith::ConstantIntOp>(loc, bitwidth, rewriter.getI32Type())
            .getResult();
    mlir::Value idConst =
        rewriter.create<arith::ConstantIntOp>(loc, id, rewriter.getI32Type())
            .getResult();
    mlir::Value intVal = rewriter
                             .create<fixedpt::BitcastOp>(
                                 loc, rewriter.getIntegerType(bitwidth), v)
                             .getResult();
    intVal = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), intVal)
                 .getResult();
    rewriter.create<LLVM::CallOp>(
        loc, printer, mlir::ValueRange{intVal, bitwidthConst, idConst});
  }

  approx::ApproxMode select_approx_mode(fixedpt::FixedPtType inputType) {
    // TODO do real heuristic
    return approx::ApproxMode::basic_poly;
  }

  mlir::Value basic_poly_approx(mlir::Value input, mlir::Value output, sollya_obj_t sollyaTree) {
    const fixedpt::FixedPtType outputType =
        output.getType().cast<fixedpt::FixedPtType>();
    const fixedpt::FixedPtType inputType =
        input.getType().cast<fixedpt::FixedPtType>();

    // We will add the rounding bit in the deg 0 coeff so we need tofixedpt::FixedPtType inputType get extra 
    // accurate to allow this rounding to be merged in the horner scheme
    const double accuracy = std::ldexp(double{1.0}, outputType.getLsb() - 2);

    /// Build sollya expresion tree
    llvm::outs() << "accuracy=" << accuracy << "\n";

    flopoco::BasicPolyApprox flopocoApprox(sollyaTree, accuracy, 0,
                                           inputType.isSigned());
    // Adding the rounding bit to the constant
    flopocoApprox.getCoeff(0)->addRoundBit(outputType.getLsb()-1);
    flopoco::VirtexUltrascalePlus flopocoTarget;
    flopoco::FixHornerArchitecture horner(
        &flopocoTarget, inputType.getLsb(), outputType.getMsb(),
        outputType.getLsb(), {&flopocoApprox});

    /// Sollya loves to change the color and not reset them
    llvm::errs().resetColor();
    llvm::outs().resetColor();

    llvm::outs() << "Polynom coefficients:" << flopocoApprox.report() << "\n";

    assert(flopocoApprox.getDegree() > 0);

    llvm::SmallVector<flopoco::FixConstant *> coefsStorage;
    for (int i = 0; i <= flopocoApprox.getDegree(); i++)
      coefsStorage.push_back(flopocoApprox.getCoeff(i));
    const llvm::ArrayRef<flopoco::FixConstant *> coefs{coefsStorage};

    printPolynom(llvm::outs(), coefs);

    // A0 + X * ( A1 + X * (...(An-1 + X * An)))
    const int printId = 0;
    std::ignore = printId;
    rewriter.setInsertionPointAfterValue(output);
    mlir::Value expr = getConstantFromCoef(coefs.back());
    int idx = horner.degree - 1;
    for (flopoco::FixConstant *flopocoCoef : llvm::reverse(coefs.drop_back())) {
      const mlir::Value coef = getConstantFromCoef(flopocoCoef);
      const mlir::Value truncatedIn = rewriter.create<fixedpt::ConvertOp>(
          loc,
          fixedpt::FixedPtType::get(ctx, inputType.getMsb(), horner.wcYLSB[idx],
                                    inputType.isSigned()),
          input, fixedpt::RoundingMode::truncate);
      fixedpt::FixedPtType mulType = inputType.getCommonMulType(
          expr.getType().cast<fixedpt::FixedPtType>());
      expr = rewriter.create<fixedpt::MulOp>(
          loc, mulType, fixedpt::RoundingMode::truncate,
          mlir::ValueRange{expr, truncatedIn});
      // emitPrintCall(expr, printId++);
      // fixedpt::FixedPtType addType =
      //     mulType.getCommonAddType(coef.getType().cast<fixedpt::FixedPtType>());
      fixedpt::FixedPtType addType = fixedpt::FixedPtType::get(
          ctx, horner.wcSumMSB[idx], horner.wcSumLSB[idx],
          true); // TODO handle the unsigned case
      expr = rewriter.create<fixedpt::AddOp>(loc, addType,
                                             fixedpt::RoundingMode::nearest,
                                             mlir::ValueRange{expr, coef});
      // emitPrintCall(expr, printId++);
      idx--;
    }

    /// This should not happened unless the result is a constant
    if (expr.getType() != outputType)
      expr = rewriter.create<fixedpt::ConvertOp>(
          loc, outputType, expr, fixedpt::RoundingMode::truncate);

    return expr;
  }

  mlir::Value getTable(std::vector<mpz_class> const & tableValues, fixedpt::FixedPtType type, mlir::Value key) {
    std::vector<mlir::Value> tableMLIRVal{};
    for (mpz_class const & val : tableValues) {
      flopoco::FixConstant cst{type.getMsb(), type.getLsb(), type.isSigned(), val};
      tableMLIRVal.push_back(getConstantFromCoef(&cst));
    }
    return rewriter.create<fixedpt::SelectOp>(loc, type, key, tableMLIRVal);
  }

  mlir::Value tabulate(mlir::Value input, mlir::Value output, sollya_obj_t sollyaTree) {
    const fixedpt::FixedPtType outputType =
        output.getType().cast<fixedpt::FixedPtType>();
    const fixedpt::FixedPtType inputType =
        input.getType().cast<fixedpt::FixedPtType>();
    flopoco::FixFunction const func{sollyaTree, inputType.isSigned(), inputType.getLsb(), outputType.getLsb()};

    if (func.msbOut > outputType.getMsb() ||
        ((func.msbOut == outputType.getMsb()) &&
         (func.signedOut != outputType.isSigned()))) {
      llvm::outs() << "Possible overflow detected in function evaluation.\n";
    }

    if (func.msbOut < outputType.getMsb()) {
      llvm::outs() << "Function evaluation result stored in an excessively wide result.\n";
    }

    mpz_class funcVal{};
    std::vector<mpz_class> tableVal;
    auto clean_overflow = [&outputType = std::as_const(outputType)](mpz_class const & to_clean){
      const auto mask = mpz_class{(mpz_class{1} << outputType.getWidth()) - 1};
      return mpz_class{to_clean & mask};
    };
    for (std::uint64_t i = 0 ; i < (uint64_t{1} << inputType.getWidth()) ; ++i) {
      mpz_class ignored{};
      func.eval(i, funcVal, ignored, true);
      tableVal.push_back(clean_overflow(funcVal));
    }

    rewriter.setInsertionPointAfterValue(output);
    return getTable(tableVal, outputType, input);
  }


  void run(approx::EvaluateOp evaluateOp) {
    loc = evaluateOp->getLoc();

    normalizeInputs(evaluateOp);

    mlir::Value output = evaluateOp.getResult();
    mlir::Value input = findInputValue(evaluateOp);
    fixedpt::FixedPtType outputType =
        output.getType().cast<fixedpt::FixedPtType>();
    fixedpt::FixedPtType inputType =
        input.getType().cast<fixedpt::FixedPtType>();

    sollya_obj_t sollyaTree = buildSollyaTree(evaluateOp);

    sollya_lib_printf("sollya: %b\n", sollyaTree);
    printFormat("input", inputType);
    printFormat("output", outputType);

    auto approx_mode = evaluateOp.getApproxMode();
    assert(approx_mode != approx::ApproxMode::bipartite_table);
    if (approx_mode == approx::ApproxMode::auto_select) {
      approx_mode = select_approx_mode(inputType);
    }

    mlir::Value expr = [&] {
      switch (approx_mode) {
      case approx::ApproxMode::basic_poly:
        return basic_poly_approx(input, output, sollyaTree);
      case approx::ApproxMode::simple_table:
        return tabulate(input, output, sollyaTree);
      default:
        llvm_unreachable("Unsupported approximation mode");
      }
    }();

    output.replaceAllUsesWith(expr);

    /// Remove the old approx tree
    llvm::SmallVector<mlir::Operation *> outdatedApproxExpr;
    approxTreeVisitor(evaluateOp, [&](mlir::Operation *op) {
      /// unlink and list everything
      op->dropAllReferences();
      outdatedApproxExpr.push_back(op);
    });
    /// erase it all
    for (auto *op : outdatedApproxExpr)
      op->erase();
  }
};

struct LowerApproxPass : approx::LowerApproxPassBase<LowerApproxPass> {
  virtual void runOnOperation() override final;
};

void LowerApproxPass::runOnOperation() {
  flopoco::set_log_lvl(flopoco::LogLevel::VERBOSE);
  LowerApprox state(getOperation()->getContext());
  if (mlir::failed(state.init()))
    signalPassFailure();
  getOperation().walk([&](approx::EvaluateOp op) { state.run(op); });
  getOperation().dump();
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::approx::createLowerApproxPass() {
  return std::make_unique<LowerApproxPass>();
}
