; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -basic-aa -newgvn -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@global = common global [1024 x i32] zeroinitializer, align 16

;; We should be able to prove the equivalence of two of the phis, and then use that to eliminate
;; one set of indexing calculations and a load

; Function Attrs: nounwind ssp uwtable
define i32 @bar(i32 %arg, i32 %arg1, i32 %arg2) #0 {
; CHECK-LABEL: @bar(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    br label %bb3
; CHECK:       bb3:
; CHECK-NEXT:    [[TMP:%.*]] = phi i32 [ %arg, %bb ], [ [[TMP:%.*]]15, %bb17 ]
; CHECK-NEXT:    [[TMP4:%.*]] = phi i32 [ %arg2, %bb ], [ [[TMP18:%.*]], %bb17 ]
; CHECK-NEXT:    [[TMP6:%.*]] = phi i32 [ 0, %bb ], [ [[TMP14:%.*]], %bb17 ]
; CHECK-NEXT:    [[TMP7:%.*]] = sext i32 [[TMP]] to i64
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds [1024 x i32], [1024 x i32]* @global, i64 0, i64 [[TMP7]]
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, i32* [[TMP8]], align 4
; CHECK-NEXT:    [[TMP10:%.*]] = add nsw i32 [[TMP6]], [[TMP9]]
; CHECK-NEXT:    [[TMP14]] = add nsw i32 [[TMP10]], [[TMP9]]
; CHECK-NEXT:    [[TMP15:%.*]] = add nsw i32 [[TMP]], %arg1
; CHECK-NEXT:    br label %bb17
; CHECK:       bb17:
; CHECK-NEXT:    [[TMP18]] = add i32 [[TMP4]], -1
; CHECK-NEXT:    [[TMP19:%.*]] = icmp ne i32 [[TMP4]], 0
; CHECK-NEXT:    br i1 [[TMP19]], label %bb3, label %bb20
; CHECK:       bb20:
; CHECK-NEXT:    ret i32 [[TMP14]]
;
bb:
  br label %bb3

bb3:                                              ; preds = %bb17, %bb
  %tmp = phi i32 [ %arg, %bb ], [ %tmp15, %bb17 ]
  %tmp4 = phi i32 [ %arg2, %bb ], [ %tmp18, %bb17 ]
  %tmp5 = phi i32 [ %arg, %bb ], [ %tmp16, %bb17 ]
  %tmp6 = phi i32 [ 0, %bb ], [ %tmp14, %bb17 ]
  %tmp7 = sext i32 %tmp to i64
  %tmp8 = getelementptr inbounds [1024 x i32], [1024 x i32]* @global, i64 0, i64 %tmp7
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = add nsw i32 %tmp6, %tmp9
  %tmp11 = sext i32 %tmp5 to i64
  %tmp12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @global, i64 0, i64 %tmp11
  %tmp13 = load i32, i32* %tmp12, align 4
  %tmp14 = add nsw i32 %tmp10, %tmp13
  %tmp15 = add nsw i32 %tmp, %arg1
  %tmp16 = add nsw i32 %tmp5, %arg1
  br label %bb17

bb17:                                             ; preds = %bb3
  %tmp18 = add i32 %tmp4, -1
  %tmp19 = icmp ne i32 %tmp4, 0
  br i1 %tmp19, label %bb3, label %bb20

bb20:                                             ; preds = %bb17
  ret i32 %tmp14
}

attributes #0 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"Apple LLVM version 8.0.0 (clang-800.0.42.1)"}
