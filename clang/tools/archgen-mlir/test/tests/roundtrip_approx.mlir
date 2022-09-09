// This tests that the format can roundtrip not what the format is.
// RUN: archgen-opt %s -o %t1.generic.mlir -mlir-print-op-generic
// RUN: archgen-opt %t1.generic.mlir -o %t2.generic.mlir -mlir-print-op-generic
// RUN: diff %t1.generic.mlir %t2.generic.mlir
// RUN: archgen-opt %s -o %t1.costum.mlir
// RUN: archgen-opt %t1.costum.mlir -o %t2.costum.mlir
// RUN: diff %t1.costum.mlir %t2.costum.mlir

module {
  func.func public @_ZN10archgenlib6detail12evaluateImplINS_11FixedNumberINS_11FixedFormatILi2ELin1EjEEEENS_5AddOpINS_5SinOpINS_8VariableINS2_INS3_ILi4ELin5EiEEEELm0EEEEENS_4PiOpEEEJSA_EEET_DpT1_(%arg0: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<2, -1, u> {
    %0 = "approx.variable"(%arg0): (!fixedpt.fixedPt<4, -5, s>) -> !approx.toBeFolded
    %1 = "approx.generic"(%0) {action = "sin"} : (!approx.toBeFolded) -> !approx.toBeFolded
    %2 = "approx.generic"() {action = "pi"} : () -> !approx.toBeFolded
    %3 = "approx.generic"(%1, %2) {action = "add"} : (!approx.toBeFolded, !approx.toBeFolded) -> !approx.toBeFolded
    %4 = "approx.evaluate"(%3) {approx_mode = 0 : i32} : (!approx.toBeFolded) -> !fixedpt.fixedPt<2, -1, u>
    return %4 : !fixedpt.fixedPt<2, -1, u>
  }
  func.func public @_ZN10archgenlib6detail12evaluateImplINS_11FixedNumberINS_11FixedFormatILi2ELin1EjEEEENS_5AddOpINS_5SinOpINS_8VariableINS2_INS3_ILi4ELin5EiEEEELm0EEEEESC_EEJSA_EEET_DpT1_(%arg0: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<2, -1, u> {
    %0 = "approx.variable"(%arg0) : (!fixedpt.fixedPt<4, -5, s>) -> !approx.toBeFolded
    %1 = "approx.generic"(%0) {action = "sin"} : (!approx.toBeFolded) -> !approx.toBeFolded
    %3 = "approx.generic"(%0) {action = "sin"} : (!approx.toBeFolded) -> !approx.toBeFolded
    %4 = "approx.generic"(%1, %3) {action = "add"} : (!approx.toBeFolded, !approx.toBeFolded) -> !approx.toBeFolded
    %5 = "approx.evaluate"(%4)  {approx_mode = 0 : i32} : (!approx.toBeFolded) -> !fixedpt.fixedPt<2, -1, u>
    return %5 : !fixedpt.fixedPt<2, -1, u>
  }
}
