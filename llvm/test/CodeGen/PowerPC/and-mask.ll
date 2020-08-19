; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; mask 0xFFFFFFFE
define i32 @test1(i32 %a) {
; CHECK-LABEL: test1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    rlwinm 3, 3, 0, 0, 30
; CHECK-NEXT:    blr
  %and = and i32 %a, -2
  ret i32 %and
}

; mask 0xFFFFFFFFFFFFFFF9
define i64 @test2(i64 %a) {
; CHECK-LABEL: test2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    rldicl 3, 3, 61, 2
; CHECK-NEXT:    rotldi 3, 3, 3
; CHECK-NEXT:    blr
  %and = and i64 %a, -7
  ret i64 %and
}

; mask: 0xFFFFFFC00000
define i64 @test3(i64 %a) {
; CHECK-LABEL: test3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    rldicl 3, 3, 42, 22
; CHECK-NEXT:    rldicl 3, 3, 22, 16
; CHECK-NEXT:    blr
  %and = and i64 %a, 281474972516352
  ret i64 %and
}

; mask: 0xC000000FF
define i64 @test4(i64 %a) {
; CHECK-LABEL: test4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    rldicl 3, 3, 30, 26
; CHECK-NEXT:    rldicl 3, 3, 34, 28
; CHECK-NEXT:    blr
  %and = and i64 %a, 51539607807
  ret i64 %and
}

; mask: 0xFFC0FFFF
define i64 @test5(i64 %a) {
; CHECK-LABEL: test5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    rldicl 3, 3, 42, 6
; CHECK-NEXT:    rldicl 3, 3, 22, 32
; CHECK-NEXT:    blr
  %and = and i64 %a, 4290838527
  ret i64 %and
}

; mask: 0x3FC0FFE0
define i64 @test6(i64 %a) {
; CHECK-LABEL: test6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lis 4, 16320
; CHECK-NEXT:    ori 4, 4, 65504
; CHECK-NEXT:    and 3, 3, 4
; CHECK-NEXT:    blr
  %and = and i64 %a, 1069613024
  ret i64 %and
}

; mask: 0x3FC000001FFFF
define i64 @test7(i64 %a) {
; CHECK-LABEL: test7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    rldicl 3, 3, 22, 25
; CHECK-NEXT:    rldicl 3, 3, 42, 14
; CHECK-NEXT:    blr
  %and = and i64 %a, 1121501860462591
  ret i64 %and
}
