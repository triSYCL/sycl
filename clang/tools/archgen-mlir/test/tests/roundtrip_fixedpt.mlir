// RUN: archgen-opt %s -o %t1.mlir
// RUN: archgen-opt %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir
// RUN: archgen-opt --convert-fixedpt-to-arith %t1.mlir

module {
  func.func public @test(%arg0: !fixedpt.fixedPt<4, -5, "signed">) -> !fixedpt.fixedPt<7, -9, "signed"> {
    %0 = "fixedpt.constant"() {valueAttr = #fixedpt.fixed_point<3, !fixedpt.fixedPt<1, -1, "unsigned">, "1.5">} : () -> !fixedpt.fixedPt<1, -1, "unsigned">
    %1 = fixedpt.round %arg0 : !fixedpt.fixedPt<4, -5, "signed"> to !fixedpt.fixedPt<4, -2, "signed">
    %2 = "fixedpt.add"(%0, %1) : (!fixedpt.fixedPt<1, -1, "unsigned">, !fixedpt.fixedPt<4, -2, "signed">) -> !fixedpt.fixedPt<4, -1, "signed">
    %3 = fixedpt.extand %2 : !fixedpt.fixedPt<4, -1, "signed"> to !fixedpt.fixedPt<7, -9, "signed">
    %4 = fixedpt.bitcast %3 : !fixedpt.fixedPt<7, -9, "signed"> as i17
    %5 = fixedpt.bitcast %4 : i17 as !fixedpt.fixedPt<7, -9, "unsigned">
    %6 = fixedpt.trunc %5 : !fixedpt.fixedPt<7, -9, "unsigned"> to !fixedpt.fixedPt<3, -9, "signed">
    %7 = "fixedpt.mul"(%1, %6) : (!fixedpt.fixedPt<4, -2, "signed">, !fixedpt.fixedPt<3, -9, "signed">) -> !fixedpt.fixedPt<7, -5, "signed">
    %8 = "fixedpt.sub"(%1, %6) : (!fixedpt.fixedPt<4, -2, "signed">, !fixedpt.fixedPt<3, -9, "signed">) -> !fixedpt.fixedPt<7, -3, "signed">
    %9 = "fixedpt.div"(%7, %8) : (!fixedpt.fixedPt<7, -5, "signed">, !fixedpt.fixedPt<7, -3, "signed">) -> !fixedpt.fixedPt<7, -9, "signed">
    return %9 : !fixedpt.fixedPt<7, -9, "signed">
  }
}
