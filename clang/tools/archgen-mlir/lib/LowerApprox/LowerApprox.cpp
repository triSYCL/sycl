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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "archgen/Approx/Approx.h"
#include "archgen/Approx/PassDetail.h"
#include "archgen/Approx/Passes.h"

#include "FixFunctions/BasicPolyApprox.hpp"

using namespace archgen;
using namespace archgen::approx;

namespace {

struct LowerApprox {
  llvm::sys::DynamicLibrary sollyaLib;

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

  template<typename T>
  T* lookupSollyaSymbol(llvm::StringRef symbol) {
    return (T*)sollyaLib.getAddressOfSymbol(symbol.data());
  }

  template <typename... Tys>
  sollya_obj_t buildSollya(llvm::StringRef name, Tys... tys) {
    llvm::Twine fullName("sollya_lib_build_function_", name);
    return lookupSollyaSymbol<sollya_obj_t(Tys...)>(fullName.str())(tys...);
  }

  sollya_obj_t buildConstant(ConstantOp op) { op->dump();
    auto fixedTy = fixedpt::FixedPtType::get(
        op->getContext(), op.valueAttr().getValue().getSemantics());
    llvm::SmallVector<char> representation;
    op.valueAttr().getValue().getValue().toString(representation, 16, false);
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

  sollya_obj_t buildSollyaTree(mlir::Operation *rawOp) {
    if (auto constant = llvm::dyn_cast<ConstantOp>(rawOp))
      return buildConstant(constant);
    genericOp op = llvm::cast<genericOp>(rawOp);
    if (op.action() == action::variable)
      return buildSollya("free_variable");
    if (op.action() == action::evaluate)
      return buildSollyaTree(op.getOperand(0).getDefiningOp());

    if (op->getNumOperands() == 0)
      return buildSollya(op.action());
    if (op->getNumOperands() == 1)
      return buildSollya(op.action(),
                         buildSollyaTree(op.getOperand(0).getDefiningOp()));
    if (op->getNumOperands() == 2)
      return buildSollya(op.action(),
                         buildSollyaTree(op.getOperand(0).getDefiningOp()),
                         buildSollyaTree(op.getOperand(1).getDefiningOp()));
    op->dump();
    llvm_unreachable("unhandeled op");
  }

  void run(genericOp evaluateOp) {
    sollya_obj_t sollyaTree = buildSollyaTree(evaluateOp);
    sollya_lib_printf("sollya: %b\n", sollyaTree);
  }
};

struct LowerApproxPass
    : LowerApproxPassBase<LowerApproxPass> {
  virtual void runOnOperation() override final;
};

void LowerApproxPass::runOnOperation() {
  LowerApprox state;
  if (mlir::failed(state.init()))
    signalPassFailure();
  getOperation().walk([&](genericOp op) {
    if (op.action() == action::evaluate)
      state.run(op);
  });
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::approx::createLowerApproxPass() {
  return std::make_unique<LowerApproxPass>();
}
