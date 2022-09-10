// RUN: archgen-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL:   func.func public @convert_fuse_add(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fixedpt.fixedPt<4, -5, s>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
// CHECK:           %[[VAL_2:.*]] = fixedpt.add %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s> to nearest <7, -9, s>
// CHECK:           return %[[VAL_2]] : !fixedpt.fixedPt<7, -9, s>
// CHECK:         }
func.func public @convert_fuse_add(%arg0: !fixedpt.fixedPt<4, -5, s>, %arg1: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
  %1 = fixedpt.add %arg0 : <4, -5, s>, %arg1 : <4, -5, s> to zero <4, -5, s>
  %2 = fixedpt.convert %1 : <4, -5, s> to nearest <7, -9, s>
  return %2 : !fixedpt.fixedPt<7, -9, s>
}

// CHECK-LABEL:   func.func public @convert_fuse_convert(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
// CHECK:           %[[VAL_1:.*]] = fixedpt.convert %[[VAL_0]] : <4, -5, s> to nearest <7, -9, s>
// CHECK:           return %[[VAL_1]] : !fixedpt.fixedPt<7, -9, s>
// CHECK:         }
func.func public @convert_fuse_convert(%arg0: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
  %1 = fixedpt.convert %arg0 : <4, -5, s> to nearest <4, -3, s>
  %2 = fixedpt.convert %1 : <4, -3, s> to zero <7, -9, s>
  return %2 : !fixedpt.fixedPt<7, -9, s>
}

// CHECK-LABEL:   func.func public @convert_fuse_round(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
// CHECK:           %[[VAL_1:.*]] = fixedpt.convert %[[VAL_0]] : <4, -5, s> to zero <7, -9, s>
// CHECK:           return %[[VAL_1]] : !fixedpt.fixedPt<7, -9, s>
// CHECK:         }
func.func public @convert_fuse_round(%arg0: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
  %1 = fixedpt.round %arg0 : <4, -5, s> to zero <4, -3, s>
  %2 = fixedpt.convert %1 : <4, -3, s> to zero <7, -9, s>
  return %2 : !fixedpt.fixedPt<7, -9, s>
}

// CHECK-LABEL:   func.func public @add_fuse_add(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fixedpt.fixedPt<4, -5, s>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
// CHECK:           %[[VAL_2:.*]] = fixedpt.add %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s>, %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s>, %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s>, %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s>, %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s>, %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -5, s> to zero <7, -9, s>
// CHECK:           return %[[VAL_2]] : !fixedpt.fixedPt<7, -9, s>
// CHECK:         }
func.func public @add_fuse_add(%arg0: !fixedpt.fixedPt<4, -5, s>, %arg1: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
  %1 = fixedpt.add %arg0 : <4, -5, s>, %arg1 : <4, -5, s> to zero <4, -5, s>
  %2 = fixedpt.add %1 : <4, -5, s>, %1 : <4, -5, s>, %1 : <4, -5, s> to zero <7, -9, s>
  %3 = fixedpt.add %2 : <7, -9, s>, %2 : <7, -9, s> to zero <7, -9, s>
  return %3 : !fixedpt.fixedPt<7, -9, s>
}

// CHECK-LABEL:   func.func public @add_fuse_constant(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
// CHECK:           %[[VAL_1:.*]] = fixedpt.constant 11 : <4, -2, u>, "2.75"
// CHECK:           %[[VAL_2:.*]] = fixedpt.add %[[VAL_0]] : <4, -5, s>, %[[VAL_1]] : <4, -2, u> to zero <7, -9, s>
// CHECK:           return %[[VAL_2]] : !fixedpt.fixedPt<7, -9, s>
// CHECK:         }
func.func public @add_fuse_constant(%arg0: !fixedpt.fixedPt<4, -5, s>) -> !fixedpt.fixedPt<7, -9, s> {
  %0 = fixedpt.constant 0 : <1, -1, u>, "0.0"
  %1 = fixedpt.constant 5 : <3, -1, u>, "2.5"
  %2 = fixedpt.constant 1 : <2, -2, u>, "0.25"
  %3 = fixedpt.add %arg0 : <4, -5, s>, %0 : <1, -1, u>, %1 : <3, -1, u>, %2 : <2, -2, u> to zero <7, -9, s>
  return %3 : !fixedpt.fixedPt<7, -9, s>
}

// CHECK-LABEL:   func.func public @add_fuse_constants2() -> !fixedpt.fixedPt<7, -9, s> {
// CHECK:           %[[VAL_0:.*]] = fixedpt.constant 1408 : <7, -9, s>, "2.75"
// CHECK:           return %[[VAL_0]] : !fixedpt.fixedPt<7, -9, s>
// CHECK:         }
func.func public @add_fuse_constants2() -> !fixedpt.fixedPt<7, -9, s> {
  %0 = fixedpt.constant 0 : <1, -1, u>, "0.0"
  %1 = fixedpt.constant 5 : <3, -1, u>, "2.5"
  %2 = fixedpt.constant 1 : <2, -2, u>, "0.25"
  %3 = fixedpt.add %0 : <1, -1, u>, %1 : <3, -1, u>, %2 : <2, -2, u> to zero <7, -9, s>
  return %3 : !fixedpt.fixedPt<7, -9, s>
}
