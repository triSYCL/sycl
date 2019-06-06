; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -reassociate %s -S -o - | FileCheck %s

; Reassociate pass used to crash on these example

@g = global i32 0

define float @undef1() {
; CHECK-LABEL: @undef1(
; CHECK-NEXT:    ret float fadd (float bitcast (i32 ptrtoint (i32* @g to i32) to float), float fadd (float bitcast (i32 ptrtoint (i32* @g to i32) to float), float fadd (float fneg (float bitcast (i32 ptrtoint (i32* @g to i32) to float)), float fneg (float bitcast (i32 ptrtoint (i32* @g to i32) to float)))))
;
  %t0 = fadd fast float bitcast (i32 ptrtoint (i32* @g to i32) to float), bitcast (i32 ptrtoint (i32* @g to i32) to float)
  %t1 = fsub fast float bitcast (i32 ptrtoint (i32* @g to i32) to float), %t0
  %t2 = fadd fast float bitcast (i32 ptrtoint (i32* @g to i32) to float), %t1
  ret float %t2
}

define void @undef2() {
; CHECK-LABEL: @undef2(
; CHECK-NEXT:    unreachable
;
  %t0 = fadd fast float bitcast (i32 ptrtoint (i32* @g to i32) to float), bitcast (i32 ptrtoint (i32* @g to i32) to float)
  %t1 = fadd fast float %t0, 1.0
  %t2 = fsub fast float %t0, %t1
  %t3 = fmul fast float %t2, 2.0
  unreachable
}

