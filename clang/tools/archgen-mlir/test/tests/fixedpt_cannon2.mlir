// RUN: archgen-opt --canonicalize --cse --convert-fixedpt-to-arith %s

module {
  func.func public @_ZN10archgenlib6detail12evaluateImplINS_11FixedNumberINS_11FixedFormatILi0ELin16EjEEEELNS_6approxE1ENS_7UnaryOpINS_8BinaryOpINS8_INS_8VariableINS2_INS3_ILi1ELin20EiEEEELm0EEENS_9NullaryOpINS0_13OperationTypeILNS_13OperationKindE10EEEEENSE_ILSF_5EEEEENS_8ConstantINS_13FixedConstantINS3_ILi2ELi0EjEELDU3_2EEEEENSE_ILSF_8EEEEENSE_ILSF_0EEEEEJSB_EEEvRT_DpRT2_(%arg0: !llvm.ptr<!fixedpt.fixedPt<0, -16, u>>, %arg1: !llvm.ptr<!fixedpt.fixedPt<1, -20, s>>) -> i8 {
    %c0_i8 = arith.constant 0 : i8
    %0 = llvm.load %arg1 : !llvm.ptr<!fixedpt.fixedPt<1, -20, s>>
    %1 = fixedpt.bitcast %0 : !fixedpt.fixedPt<1, -20, s> as !fixedpt.fixedPt<0, -21, s>
    %2 = fixedpt.constant -101 : <-7, -14, s>, "-0.00616455078125"
    %3 = fixedpt.constant 0 : <1, 0, s>, "0.0"
    %4 = fixedpt.mul %2 : <-7, -14, s>, %1 : <0, -21, s> truncate <-6, -35, s>
    %5 = fixedpt.add %4 : <-6, -35, s>, %3 : <1, 0, s> nearest <-7, -21, s>
    %6 = fixedpt.constant 1328 : <-3, -14, s>, "0.0810546875"
    %7 = fixedpt.mul %5 : <-7, -21, s>, %1 : <0, -21, s> truncate <-6, -42, s>
    %8 = fixedpt.add %7 : <-6, -42, s>, %6 : <-3, -14, s> nearest <-3, -21, u>
    %9 = fixedpt.constant 0 : <1, 0, s>, "0.0"
    %10 = fixedpt.mul %8 : <-3, -21, u>, %1 : <0, -21, s> truncate <-2, -42, s>
    %11 = fixedpt.add %10 : <-2, -42, s>, %9 : <1, 0, s> nearest <-3, -21, s>
    %12 = fixedpt.constant -9814 : <0, -14, s>, "-0.5989990234375"
    %13 = fixedpt.mul %11 : <-3, -21, s>, %1 : <0, -21, s> truncate <-2, -42, s>
    %14 = fixedpt.add %13 : <-2, -42, s>, %12 : <0, -14, s> nearest <0, -21, u>
    %15 = fixedpt.constant 0 : <1, 0, s>, "0.0"
    %16 = fixedpt.mul %14 : <0, -21, u>, %1 : <0, -21, s> truncate <1, -42, s>
    %17 = fixedpt.add %16 : <1, -42, s>, %15 : <1, 0, s> nearest <0, -21, s>
    %18 = fixedpt.constant 41784 : <2, -14, s>, "2.55029296875"
    %19 = fixedpt.mul %17 : <0, -21, s>, %1 : <0, -21, s> truncate <1, -42, s>
    %20 = fixedpt.add %19 : <1, -42, s>, %18 : <2, -14, s> nearest <2, -21, u>
    %21 = fixedpt.constant 0 : <1, 0, s>, "0.0"
    %22 = fixedpt.mul %20 : <2, -21, u>, %1 : <0, -21, s> truncate <3, -42, s>
    %23 = fixedpt.add %22 : <3, -42, s>, %21 : <1, 0, s> nearest <2, -21, s>
    %24 = fixedpt.constant -84669 : <3, -14, s>, "-5.16778564453125"
    %25 = fixedpt.mul %23 : <2, -21, s>, %1 : <0, -21, s> truncate <3, -42, s>
    %26 = fixedpt.add %25 : <3, -42, s>, %24 : <3, -14, s> nearest <3, -21, u>
    %27 = fixedpt.constant 0 : <1, 0, s>, "0.0"
    %28 = fixedpt.mul %26 : <3, -21, u>, %1 : <0, -21, s> truncate <4, -42, s>
    %29 = fixedpt.add %28 : <4, -42, s>, %27 : <1, 0, s> nearest <3, -21, s>
    %30 = fixedpt.constant 51472 : <2, -14, s>, "3.1416015625"
    %31 = fixedpt.mul %29 : <3, -21, s>, %1 : <0, -21, s> truncate <4, -42, s>
    %32 = fixedpt.add %31 : <4, -42, s>, %30 : <2, -14, s> nearest <2, -21, s>
    %33 = fixedpt.constant 1 : <-16, -17, s>, "0.00000762939453125"
    %34 = fixedpt.mul %32 : <2, -21, s>, %1 : <0, -21, s> truncate <3, -42, s>
    %35 = fixedpt.add %34 : <3, -42, s>, %33 : <-16, -17, s> nearest <2, -21, s>
    %36 = fixedpt.convert %35 : <2, -21, s> truncate <0, -16, u>
    llvm.store %36, %arg0 : !llvm.ptr<!fixedpt.fixedPt<0, -16, u>>
    return %c0_i8 : i8
  }
}
