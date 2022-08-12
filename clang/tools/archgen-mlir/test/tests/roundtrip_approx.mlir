// RUN: archgen-opt %s -o %t1.mlir
// RUN: archgen-opt %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

module {
  func.func public @_ZN10archgenlib6detail12evaluateImplINS_11FixedNumberINS_11FixedFormatILi2ELin1EjEEEENS_5AddOpINS_5SinOpINS_8VariableINS2_INS3_ILi4ELin5EiEEEELm0EEEEENS_4PiOpEEEJSA_EEET_DpT1_(%arg0: !fixedpt.fixedPt<4, -5, "signed">) -> !fixedpt.fixedPt<2, -1, "unsigned"> {
    %0 = "approx.generic"(%arg0) {action = "variable"} : (!fixedpt.fixedPt<4, -5, "signed">) -> !approx.toBeFolded
    %1 = "approx.generic"(%0) {action = "sin"} : (!approx.toBeFolded) -> !approx.toBeFolded
    %2 = "approx.generic"() {action = "pi"} : () -> !approx.toBeFolded
    %3 = "approx.generic"(%1, %2) {action = "add"} : (!approx.toBeFolded, !approx.toBeFolded) -> !approx.toBeFolded
    %4 = "approx.generic"(%3) {action = "evaluate"} : (!approx.toBeFolded) -> !fixedpt.fixedPt<2, -1, "unsigned">
    return %4 : !fixedpt.fixedPt<2, -1, "unsigned">
  }
  func.func public @_ZN10archgenlib6detail12evaluateImplINS_11FixedNumberINS_11FixedFormatILi2ELin1EjEEEENS_5AddOpINS_5SinOpINS_8VariableINS2_INS3_ILi4ELin5EiEEEELm0EEEEESC_EEJSA_EEET_DpT1_(%arg0: !fixedpt.fixedPt<4, -5, "signed">) -> !fixedpt.fixedPt<2, -1, "unsigned"> {
    %0 = "approx.generic"(%arg0) {action = "variable"} : (!fixedpt.fixedPt<4, -5, "signed">) -> !approx.toBeFolded
    %1 = "approx.generic"(%0) {action = "sin"} : (!approx.toBeFolded) -> !approx.toBeFolded
    %2 = "approx.generic"(%arg0) {action = "variable"} : (!fixedpt.fixedPt<4, -5, "signed">) -> !approx.toBeFolded
    %3 = "approx.generic"(%2) {action = "sin"} : (!approx.toBeFolded) -> !approx.toBeFolded
    %4 = "approx.generic"(%1, %3) {action = "add"} : (!approx.toBeFolded, !approx.toBeFolded) -> !approx.toBeFolded
    %5 = "approx.generic"(%4) {action = "evaluate"} : (!approx.toBeFolded) -> !fixedpt.fixedPt<2, -1, "unsigned">
    return %5 : !fixedpt.fixedPt<2, -1, "unsigned">
  }
}
