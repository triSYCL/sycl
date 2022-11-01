// This tests that the format can roundtrip not what the format is.
// RUN: archgen-opt %s -o %t1.generic.mlir -mlir-print-op-generic
// RUN: archgen-opt %t1.generic.mlir -o %t2.generic.mlir -mlir-print-op-generic
// RUN: diff %t1.generic.mlir %t2.generic.mlir
// RUN: archgen-opt %s -o %t1.costum.mlir
// RUN: archgen-opt %t1.costum.mlir -o %t2.costum.mlir
// RUN: diff %t1.costum.mlir %t2.costum.mlir
// RUN: archgen-opt --convert-fixedpt-to-arith %t1.costum.mlir

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !fixedpt.fixedPt<4, -5, s>):
    %0 = "fixedpt.constant"() {valueAttr = #fixedpt.fixed_point<3, !fixedpt.fixedPt<1, -1, u>, "1.5">} : () -> !fixedpt.fixedPt<1, -1, u>
    %1 = "fixedpt.round"(%arg0) {rounding = 0 : i32} : (!fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<4, -2, s>
    %2 = "fixedpt.add"(%0, %1, %0, %0) {rounding = 1 : i32} : (!fixedpt.fixedPt<1, -1, u>, !fixedpt.fixedPt<4, -2, s>, !fixedpt.fixedPt<1, -1, u>, !fixedpt.fixedPt<1, -1, u>) -> !fixedpt.fixedPt<4, -1, s>
    %10 = "fixedpt.mul"(%0, %1, %0, %0) {rounding = 2 : i32} : (!fixedpt.fixedPt<1, -1, u>, !fixedpt.fixedPt<4, -2, s>, !fixedpt.fixedPt<1, -1, u>, !fixedpt.fixedPt<1, -1, u>) -> !fixedpt.fixedPt<4, -1, s>
    %11 = "fixedpt.mul"(%0, %1, %0, %0) {rounding = 3 : i32} : (!fixedpt.fixedPt<1, -1, u>, !fixedpt.fixedPt<4, -2, s>, !fixedpt.fixedPt<1, -1, u>, !fixedpt.fixedPt<1, -1, u>) -> !fixedpt.fixedPt<4, -1, s>
    %3 = "fixedpt.extend"(%2) : (!fixedpt.fixedPt<4, -1, s>) -> !fixedpt.fixedPt<7, -9, s>
    %4 = "fixedpt.bitcast"(%3) : (!fixedpt.fixedPt<7, -9, s>) -> i17
    %5 = "fixedpt.bitcast"(%4) : (i17) -> !fixedpt.fixedPt<7, -9, u>
    %6 = "fixedpt.trunc"(%5) : (!fixedpt.fixedPt<7, -9, u>) -> !fixedpt.fixedPt<3, -9, s>
    %7 = "fixedpt.mul"(%1, %6) {rounding = 1 : i32} : (!fixedpt.fixedPt<4, -2, s>, !fixedpt.fixedPt<3, -9, s>) -> !fixedpt.fixedPt<7, -5, s>
    %8 = "fixedpt.sub"(%1, %6) {rounding = 1 : i32} : (!fixedpt.fixedPt<4, -2, s>, !fixedpt.fixedPt<3, -9, s>) -> !fixedpt.fixedPt<7, -3, s>
    %9 = "fixedpt.div"(%7, %8) {rounding = 1 : i32} : (!fixedpt.fixedPt<7, -5, s>, !fixedpt.fixedPt<7, -3, s>) -> !fixedpt.fixedPt<7, -9, s>
    "func.return"(%9) : (!fixedpt.fixedPt<7, -9, s>) -> ()
  }) {function_type = (!fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s>, sym_name = "test", sym_visibility = "public"} : () -> ()
}) : () -> ()
