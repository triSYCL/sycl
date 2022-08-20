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

#include <thread>
#include <chrono>

#include "llvm/Support/DynamicLibrary.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "archgen/Approx/Approx.h"
#include "archgen/Approx/PassDetail.h"
#include "archgen/Approx/Passes.h"

#include "flopoco/FixFunctions/BasicPolyApprox.hpp"

#ifdef __linux__
#include <unistd.h>
#include <signal.h>

struct KillableOnInput {
  std::thread t;
  std::atomic<bool> shouldStop{false};
  KillableOnInput() {
    llvm::errs() << "Sollya is intercepting SIGINT, press enter to kill the process\n";
    llvm::errs().flush();
    t = std::thread([&] { background(); });
  }
  void stop() {
    shouldStop = true;
    t.join();
    ::signal(SIGINT, SIG_DFL);
    llvm::errs() << "SIGINT back to normal\n";
  }
  ~KillableOnInput() {
    if (t.joinable() && !shouldStop)
      stop();
  }

private:
  void terminate() {
    llvm::errs().resetColor();
    llvm::outs().resetColor();
    std::terminate();
  }
  void background() {
    fd_set set;
    timeval timeout;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    timeout.tv_sec = 0;
    timeout.tv_usec = 100;

    while (!shouldStop) {
      int res = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout);
      if (res != 0) // error or data to read
        terminate();
    }
  }
};
#else
struct KillableOnInput {
  void stop() {}
};
#endif

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
            return op.output();

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
                         .result();

          /// This is the new input to the expression
          op.setOperand(variable);

          /// build the scaling constant
          rewriter.setInsertionPointAfter(op);
          mlir::Value scalingFactor =
              rewriter
                  .create<approx::ConstantOp>(
                      op.getLoc(), fixedpt::FixedPointAttr::get(
                                       ctx, std::move(fixedScalingFactor)))
                  .result();

          /// build the multiply
          return rewriter
              .create<approx::GenericOp>(op->getLoc(),
                                 mlir::ValueRange{op.output(), scalingFactor},
                                 "mul")
              .output();
        });
  }

  mlir::Value findInputValue(approx::EvaluateOp evaluateOp) {
    mlir::Value inputVal;
    approxTreeVisitor(evaluateOp, [&](mlir::Operation *op) {
      if (auto var = llvm::dyn_cast<approx::VariableOp>(op)) {
        assert(!inputVal || (inputVal == var.input()));
        inputVal = var.input();
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
          return buildSollya(op.action(), operands...);
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

  mlir::Value convertTo(mlir::Value in, fixedpt::FixedPtType outTy) {
    if (in.getType() == outTy)
      return in;

    /// it is easier to to just build "x + 0" and let lowering to arith
    /// do the conversion. we really should have a simple convertOp. its easy to
    /// implement

    /// all bits to zero is zero for all formats
    llvm::APFixedPoint zeroFP(
        0, llvm::FixedPointSemantics(1, 0, false, false, false));
    fixedpt::FixedPtType zeroTy =
        fixedpt::FixedPtType::get(ctx, zeroFP.getSemantics());

    mlir::Value zero =
        rewriter
            .create<fixedpt::ConstantOp>(
                loc, zeroTy,
                fixedpt::FixedPointAttr::get(ctx, std::move(zeroFP)))
            .result();
    return rewriter.create<fixedpt::AddOp>(loc, outTy, in, zero);
  }

  void run(approx::EvaluateOp evaluateOp) {
    loc = evaluateOp->getLoc();

    normalizeInputs(evaluateOp);

    mlir::Value output = evaluateOp.getResult();
    mlir::Value input = findInputValue(evaluateOp);
    fixedpt::FixedPtType outputType = output.getType().cast<fixedpt::FixedPtType>();
    fixedpt::FixedPtType inputType = input.getType().cast<fixedpt::FixedPtType>();

    sollya_obj_t sollyaTree = buildSollyaTree(evaluateOp);
    double accuracy = std::pow(2.0, outputType.getLsb() - 1);

    /// Build sollya expresion tree
    sollya_lib_printf("sollya: %b\n", sollyaTree);
    printFormat("input", inputType);
    printFormat("output", outputType);
    llvm::outs() << "accuracy="  << accuracy << "\n";

    KillableOnInput koi;
    flopoco::BasicPolyApprox flopocoApprox(sollyaTree, accuracy, 0,
                                           inputType.isSigned());
    koi.stop();

    /// Sollya loves to change the color and not reset them
    llvm::errs().resetColor();
    llvm::outs().resetColor();

    assert(flopocoApprox.getDegree() > 0);

    llvm::SmallVector<flopoco::FixConstant*> coefsStorage;
    for (int i = 0; i < flopocoApprox.getDegree(); i++)
      coefsStorage.push_back(flopocoApprox.getCoeff(i));
    llvm::ArrayRef<flopoco::FixConstant*> coefs{coefsStorage};

    printPolynom(llvm::outs(), coefs);

    // A0 + X * ( A1 + X * (...(An-1 + X * An)))
    rewriter.setInsertionPointAfterValue(output);
    mlir::Value expr = getConstantFromCoef(coefs.back());
    for (flopoco::FixConstant* flopocoCoef : llvm::reverse(coefs.drop_back())) {
      mlir::Value coef = getConstantFromCoef(flopocoCoef);
      fixedpt::FixedPtType addType = inputType.getCommonAddType(
          coef.getType().cast<fixedpt::FixedPtType>());
      mlir::Value AplusX =
          rewriter.create<fixedpt::AddOp>(loc, addType, input, coef).result();
      fixedpt::FixedPtType mulType =
          addType.getCommonMulType(expr.getType().cast<fixedpt::FixedPtType>());
      expr =
          rewriter.create<fixedpt::MulOp>(loc, mulType, expr, AplusX).result();
    }

    /// This should not happened unless the result is a constant
    if (expr.getType() != outputType)
      expr = convertTo(expr, outputType);
    output.replaceAllUsesWith(expr);

    /// Remove the old approx tree
    llvm::SmallVector<mlir::Operation*> outdatedApproxExpr;
    approxTreeVisitor(evaluateOp, [&](mlir::Operation *op) {
      /// unlink and list everything
      op->dropAllReferences();
      outdatedApproxExpr.push_back(op);
    });
    /// erase it all
    for (auto* op : outdatedApproxExpr)
      op->erase();

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
