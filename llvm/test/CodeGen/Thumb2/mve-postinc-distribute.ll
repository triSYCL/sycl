; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve.fp %s -o - | FileCheck %s

; Check some loop postinc's for properly distributed post-incs

define i32 @vaddv(i32* nocapture readonly %data, i32 %N) {
; CHECK-LABEL: vaddv:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    mov lr, r1
; CHECK-NEXT:    cmp r1, #1
; CHECK-NEXT:    blt .LBB0_4
; CHECK-NEXT:  @ %bb.1: @ %for.body.preheader
; CHECK-NEXT:    mov r1, r0
; CHECK-NEXT:    movs r0, #0
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB0_2: @ %for.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrw.u32 q0, [r1], #32
; CHECK-NEXT:    vaddva.s32 r0, q0
; CHECK-NEXT:    vldrw.u32 q0, [r1, #-16]
; CHECK-NEXT:    vaddva.s32 r0, q0
; CHECK-NEXT:    le lr, .LBB0_2
; CHECK-NEXT:  @ %bb.3: @ %for.cond.cleanup
; CHECK-NEXT:    pop {r7, pc}
; CHECK-NEXT:  .LBB0_4:
; CHECK-NEXT:    movs r0, #0
; CHECK-NEXT:    pop {r7, pc}
entry:
  %cmp11 = icmp sgt i32 %N, 0
  br i1 %cmp11, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %x.0.lcssa = phi i32 [ 0, %entry ], [ %7, %for.body ]
  ret i32 %x.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %data.addr.014 = phi i32* [ %add.ptr1, %for.body ], [ %data, %entry ]
  %i.013 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %x.012 = phi i32 [ %7, %for.body ], [ 0, %entry ]
  %0 = bitcast i32* %data.addr.014 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 4
  %2 = tail call i32 @llvm.arm.mve.addv.v4i32(<4 x i32> %1, i32 0)
  %3 = add i32 %2, %x.012
  %add.ptr = getelementptr inbounds i32, i32* %data.addr.014, i32 4
  %4 = bitcast i32* %add.ptr to <4 x i32>*
  %5 = load <4 x i32>, <4 x i32>* %4, align 4
  %6 = tail call i32 @llvm.arm.mve.addv.v4i32(<4 x i32> %5, i32 0)
  %7 = add i32 %3, %6
  %add.ptr1 = getelementptr inbounds i32, i32* %data.addr.014, i32 8
  %inc = add nuw nsw i32 %i.013, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @arm_cmplx_dot_prod_q15(i16* nocapture readonly %pSrcA, i16* nocapture readonly %pSrcB, i32 %numSamples, i32* nocapture %realResult, i32* nocapture %imagResult) {
; CHECK-LABEL: arm_cmplx_dot_prod_q15:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    mvn r7, #7
; CHECK-NEXT:    mov.w r12, #0
; CHECK-NEXT:    add.w r7, r7, r2, lsl #1
; CHECK-NEXT:    vldrh.u16 q0, [r0]
; CHECK-NEXT:    vldrh.u16 q1, [r1]
; CHECK-NEXT:    movs r4, #0
; CHECK-NEXT:    lsr.w lr, r7, #3
; CHECK-NEXT:    mov r7, r12
; CHECK-NEXT:    mov r11, r12
; CHECK-NEXT:    wls lr, lr, .LBB1_4
; CHECK-NEXT:  @ %bb.1: @ %while.body.preheader
; CHECK-NEXT:    mov.w r11, #0
; CHECK-NEXT:    add.w r8, r0, lr, lsl #5
; CHECK-NEXT:    adds r0, #32
; CHECK-NEXT:    add.w r6, r1, #32
; CHECK-NEXT:    lsl.w r9, lr, #4
; CHECK-NEXT:    mov r4, r11
; CHECK-NEXT:    movs r7, #0
; CHECK-NEXT:    mov r12, r11
; CHECK-NEXT:  .LBB1_2: @ %while.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrh.u16 q2, [r6, #-16]
; CHECK-NEXT:    vldrh.u16 q3, [r0, #-16]
; CHECK-NEXT:    vmlaldavax.s16 r4, r11, q0, q1
; CHECK-NEXT:    vmlsldava.s16 r12, r7, q0, q1
; CHECK-NEXT:    vldrh.u16 q0, [r0], #32
; CHECK-NEXT:    vldrh.u16 q1, [r6], #32
; CHECK-NEXT:    vmlaldavax.s16 r4, r11, q3, q2
; CHECK-NEXT:    vmlsldava.s16 r12, r7, q3, q2
; CHECK-NEXT:    le lr, .LBB1_2
; CHECK-NEXT:  @ %bb.3: @ %while.cond.while.end_crit_edge
; CHECK-NEXT:    add.w r1, r1, r9, lsl #1
; CHECK-NEXT:    mov r0, r8
; CHECK-NEXT:  .LBB1_4: @ %while.end
; CHECK-NEXT:    vmlaldavax.s16 r4, r11, q0, q1
; CHECK-NEXT:    vmlsldava.s16 r12, r7, q0, q1
; CHECK-NEXT:    mov r10, r4
; CHECK-NEXT:    mov r5, r11
; CHECK-NEXT:    lsrl r10, r5, #6
; CHECK-NEXT:    ldr.w r8, [sp, #36]
; CHECK-NEXT:    mov r6, r12
; CHECK-NEXT:    mov r5, r7
; CHECK-NEXT:    and lr, r2, #3
; CHECK-NEXT:    lsrl r6, r5, #6
; CHECK-NEXT:    wls lr, lr, .LBB1_7
; CHECK-NEXT:  .LBB1_5: @ %while.body11
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    ldrsh.w r5, [r0, #2]
; CHECK-NEXT:    ldrsh.w r6, [r1]
; CHECK-NEXT:    ldrsh.w r9, [r0]
; CHECK-NEXT:    adds r0, #4
; CHECK-NEXT:    ldrsh.w r2, [r1, #2]
; CHECK-NEXT:    adds r1, #4
; CHECK-NEXT:    smlalbb r4, r11, r6, r5
; CHECK-NEXT:    smlalbb r12, r7, r6, r9
; CHECK-NEXT:    muls r5, r2, r5
; CHECK-NEXT:    smlalbb r4, r11, r2, r9
; CHECK-NEXT:    subs.w r12, r12, r5
; CHECK-NEXT:    sbc.w r7, r7, r5, asr #31
; CHECK-NEXT:    le lr, .LBB1_5
; CHECK-NEXT:  @ %bb.6: @ %while.end34.loopexit
; CHECK-NEXT:    lsrl r12, r7, #6
; CHECK-NEXT:    lsrl r4, r11, #6
; CHECK-NEXT:    mov r6, r12
; CHECK-NEXT:    mov r10, r4
; CHECK-NEXT:  .LBB1_7: @ %while.end34
; CHECK-NEXT:    str r6, [r3]
; CHECK-NEXT:    str.w r10, [r8]
; CHECK-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
entry:
  %mul = shl i32 %numSamples, 1
  %sub = add i32 %mul, -8
  %shr = lshr i32 %sub, 3
  %vecSrcB.0.in102 = bitcast i16* %pSrcB to <8 x i16>*
  %vecSrcB.0103 = load <8 x i16>, <8 x i16>* %vecSrcB.0.in102, align 2
  %vecSrcA.0.in104 = bitcast i16* %pSrcA to <8 x i16>*
  %vecSrcA.0105 = load <8 x i16>, <8 x i16>* %vecSrcA.0.in104, align 2
  %cmp106 = icmp eq i32 %shr, 0
  br i1 %cmp106, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  %0 = shl i32 %shr, 4
  %scevgep = getelementptr i16, i16* %pSrcA, i32 %0
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %vecSrcA.0115 = phi <8 x i16> [ %vecSrcA.0, %while.body ], [ %vecSrcA.0105, %while.body.preheader ]
  %vecSrcB.0114 = phi <8 x i16> [ %vecSrcB.0, %while.body ], [ %vecSrcB.0103, %while.body.preheader ]
  %vecSrcB.0.in.in113 = phi i16* [ %add.ptr3, %while.body ], [ %pSrcB, %while.body.preheader ]
  %vecSrcA.0.in.in112 = phi i16* [ %add.ptr2, %while.body ], [ %pSrcA, %while.body.preheader ]
  %accImag.0.off32111 = phi i32 [ %15, %while.body ], [ 0, %while.body.preheader ]
  %accImag.0.off0110 = phi i32 [ %16, %while.body ], [ 0, %while.body.preheader ]
  %accReal.0.off32109 = phi i32 [ %12, %while.body ], [ 0, %while.body.preheader ]
  %accReal.0.off0108 = phi i32 [ %13, %while.body ], [ 0, %while.body.preheader ]
  %blkCnt.0107 = phi i32 [ %dec, %while.body ], [ %shr, %while.body.preheader ]
  %pSrcB.addr.0 = getelementptr inbounds i16, i16* %vecSrcB.0.in.in113, i32 8
  %pSrcA.addr.0 = getelementptr inbounds i16, i16* %vecSrcA.0.in.in112, i32 8
  %1 = tail call { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32 0, i32 1, i32 0, i32 %accReal.0.off0108, i32 %accReal.0.off32109, <8 x i16> %vecSrcA.0115, <8 x i16> %vecSrcB.0114)
  %2 = extractvalue { i32, i32 } %1, 1
  %3 = extractvalue { i32, i32 } %1, 0
  %4 = bitcast i16* %pSrcA.addr.0 to <8 x i16>*
  %5 = load <8 x i16>, <8 x i16>* %4, align 2
  %add.ptr2 = getelementptr inbounds i16, i16* %vecSrcA.0.in.in112, i32 16
  %6 = tail call { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32 0, i32 0, i32 1, i32 %accImag.0.off0110, i32 %accImag.0.off32111, <8 x i16> %vecSrcA.0115, <8 x i16> %vecSrcB.0114)
  %7 = extractvalue { i32, i32 } %6, 1
  %8 = extractvalue { i32, i32 } %6, 0
  %9 = bitcast i16* %pSrcB.addr.0 to <8 x i16>*
  %10 = load <8 x i16>, <8 x i16>* %9, align 2
  %add.ptr3 = getelementptr inbounds i16, i16* %vecSrcB.0.in.in113, i32 16
  %11 = tail call { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32 0, i32 1, i32 0, i32 %3, i32 %2, <8 x i16> %5, <8 x i16> %10)
  %12 = extractvalue { i32, i32 } %11, 1
  %13 = extractvalue { i32, i32 } %11, 0
  %14 = tail call { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32 0, i32 0, i32 1, i32 %8, i32 %7, <8 x i16> %5, <8 x i16> %10)
  %15 = extractvalue { i32, i32 } %14, 1
  %16 = extractvalue { i32, i32 } %14, 0
  %dec = add nsw i32 %blkCnt.0107, -1
  %vecSrcB.0.in = bitcast i16* %add.ptr3 to <8 x i16>*
  %vecSrcB.0 = load <8 x i16>, <8 x i16>* %vecSrcB.0.in, align 2
  %vecSrcA.0.in = bitcast i16* %add.ptr2 to <8 x i16>*
  %vecSrcA.0 = load <8 x i16>, <8 x i16>* %vecSrcA.0.in, align 2
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.cond.while.end_crit_edge, label %while.body

while.cond.while.end_crit_edge:                   ; preds = %while.body
  %scevgep136 = getelementptr i16, i16* %pSrcB, i32 %0
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  %accReal.0.off0.lcssa = phi i32 [ %13, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  %accReal.0.off32.lcssa = phi i32 [ %12, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  %accImag.0.off0.lcssa = phi i32 [ %16, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  %accImag.0.off32.lcssa = phi i32 [ %15, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  %vecSrcA.0.in.in.lcssa = phi i16* [ %scevgep, %while.cond.while.end_crit_edge ], [ %pSrcA, %entry ]
  %vecSrcB.0.in.in.lcssa = phi i16* [ %scevgep136, %while.cond.while.end_crit_edge ], [ %pSrcB, %entry ]
  %vecSrcB.0.lcssa = phi <8 x i16> [ %vecSrcB.0, %while.cond.while.end_crit_edge ], [ %vecSrcB.0103, %entry ]
  %vecSrcA.0.lcssa = phi <8 x i16> [ %vecSrcA.0, %while.cond.while.end_crit_edge ], [ %vecSrcA.0105, %entry ]
  %17 = tail call { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32 0, i32 1, i32 0, i32 %accReal.0.off0.lcssa, i32 %accReal.0.off32.lcssa, <8 x i16> %vecSrcA.0.lcssa, <8 x i16> %vecSrcB.0.lcssa)
  %18 = extractvalue { i32, i32 } %17, 1
  %19 = zext i32 %18 to i64
  %20 = shl nuw i64 %19, 32
  %21 = extractvalue { i32, i32 } %17, 0
  %22 = zext i32 %21 to i64
  %23 = or i64 %20, %22
  %24 = tail call { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32 0, i32 0, i32 1, i32 %accImag.0.off0.lcssa, i32 %accImag.0.off32.lcssa, <8 x i16> %vecSrcA.0.lcssa, <8 x i16> %vecSrcB.0.lcssa)
  %25 = extractvalue { i32, i32 } %24, 1
  %26 = zext i32 %25 to i64
  %27 = shl nuw i64 %26, 32
  %28 = extractvalue { i32, i32 } %24, 0
  %29 = zext i32 %28 to i64
  %30 = or i64 %27, %29
  %shr8 = and i32 %numSamples, 3
  %cmp1095 = icmp eq i32 %shr8, 0
  %extract = lshr i64 %23, 6
  %extract.t = trunc i64 %extract to i32
  %extract129 = lshr i64 %30, 6
  %extract.t130 = trunc i64 %extract129 to i32
  br i1 %cmp1095, label %while.end34, label %while.body11

while.body11:                                     ; preds = %while.end, %while.body11
  %pSrcA.addr.1100 = phi i16* [ %incdec.ptr12, %while.body11 ], [ %vecSrcA.0.in.in.lcssa, %while.end ]
  %pSrcB.addr.199 = phi i16* [ %incdec.ptr14, %while.body11 ], [ %vecSrcB.0.in.in.lcssa, %while.end ]
  %accImag.198 = phi i64 [ %add32, %while.body11 ], [ %30, %while.end ]
  %accReal.197 = phi i64 [ %sub27, %while.body11 ], [ %23, %while.end ]
  %blkCnt.196 = phi i32 [ %dec33, %while.body11 ], [ %shr8, %while.end ]
  %incdec.ptr = getelementptr inbounds i16, i16* %pSrcA.addr.1100, i32 1
  %31 = load i16, i16* %pSrcA.addr.1100, align 2
  %incdec.ptr12 = getelementptr inbounds i16, i16* %pSrcA.addr.1100, i32 2
  %32 = load i16, i16* %incdec.ptr, align 2
  %incdec.ptr13 = getelementptr inbounds i16, i16* %pSrcB.addr.199, i32 1
  %33 = load i16, i16* %pSrcB.addr.199, align 2
  %incdec.ptr14 = getelementptr inbounds i16, i16* %pSrcB.addr.199, i32 2
  %34 = load i16, i16* %incdec.ptr13, align 2
  %conv = sext i16 %31 to i32
  %conv15 = sext i16 %33 to i32
  %mul16 = mul nsw i32 %conv15, %conv
  %conv17 = sext i32 %mul16 to i64
  %add = add nsw i64 %accReal.197, %conv17
  %conv19 = sext i16 %34 to i32
  %mul20 = mul nsw i32 %conv19, %conv
  %conv21 = sext i32 %mul20 to i64
  %conv23 = sext i16 %32 to i32
  %mul25 = mul nsw i32 %conv19, %conv23
  %conv26 = sext i32 %mul25 to i64
  %sub27 = sub i64 %add, %conv26
  %mul30 = mul nsw i32 %conv15, %conv23
  %conv31 = sext i32 %mul30 to i64
  %add22 = add i64 %accImag.198, %conv31
  %add32 = add i64 %add22, %conv21
  %dec33 = add nsw i32 %blkCnt.196, -1
  %cmp10 = icmp eq i32 %dec33, 0
  br i1 %cmp10, label %while.end34.loopexit, label %while.body11

while.end34.loopexit:                             ; preds = %while.body11
  %extract131 = lshr i64 %add32, 6
  %extract.t132 = trunc i64 %extract131 to i32
  %extract127 = lshr i64 %sub27, 6
  %extract.t128 = trunc i64 %extract127 to i32
  br label %while.end34

while.end34:                                      ; preds = %while.end34.loopexit, %while.end
  %accReal.1.lcssa.off6 = phi i32 [ %extract.t, %while.end ], [ %extract.t128, %while.end34.loopexit ]
  %accImag.1.lcssa.off6 = phi i32 [ %extract.t130, %while.end ], [ %extract.t132, %while.end34.loopexit ]
  store i32 %accReal.1.lcssa.off6, i32* %realResult, align 4
  store i32 %accImag.1.lcssa.off6, i32* %imagResult, align 4
  ret void
}


define void @fma8(float* noalias nocapture readonly %A, float* noalias nocapture readonly %B, float* noalias nocapture %C, i32 %n) {
; CHECK-LABEL: fma8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, lr}
; CHECK-NEXT:    push {r4, r5, r6, lr}
; CHECK-NEXT:    cmp r3, #1
; CHECK-NEXT:    blt .LBB2_8
; CHECK-NEXT:  @ %bb.1: @ %for.body.preheader
; CHECK-NEXT:    cmp r3, #7
; CHECK-NEXT:    bhi .LBB2_3
; CHECK-NEXT:  @ %bb.2:
; CHECK-NEXT:    mov.w r12, #0
; CHECK-NEXT:    b .LBB2_6
; CHECK-NEXT:  .LBB2_3: @ %vector.ph
; CHECK-NEXT:    bic r12, r3, #7
; CHECK-NEXT:    movs r5, #1
; CHECK-NEXT:    sub.w r6, r12, #8
; CHECK-NEXT:    mov r4, r0
; CHECK-NEXT:    add.w lr, r5, r6, lsr #3
; CHECK-NEXT:    mov r5, r1
; CHECK-NEXT:    mov r6, r2
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB2_4: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrw.u32 q0, [r4, #16]
; CHECK-NEXT:    vldrw.u32 q1, [r5, #16]
; CHECK-NEXT:    vldrw.u32 q2, [r6, #16]
; CHECK-NEXT:    vldrw.u32 q3, [r6]
; CHECK-NEXT:    vfma.f32 q2, q1, q0
; CHECK-NEXT:    vldrw.u32 q0, [r4], #32
; CHECK-NEXT:    vldrw.u32 q1, [r5], #32
; CHECK-NEXT:    vfma.f32 q3, q1, q0
; CHECK-NEXT:    vstrw.32 q3, [r6], #32
; CHECK-NEXT:    vstrw.32 q2, [r6, #-16]
; CHECK-NEXT:    le lr, .LBB2_4
; CHECK-NEXT:  @ %bb.5: @ %middle.block
; CHECK-NEXT:    cmp r12, r3
; CHECK-NEXT:    it eq
; CHECK-NEXT:    popeq {r4, r5, r6, pc}
; CHECK-NEXT:  .LBB2_6: @ %for.body.preheader12
; CHECK-NEXT:    sub.w lr, r3, r12
; CHECK-NEXT:    add.w r0, r0, r12, lsl #2
; CHECK-NEXT:    add.w r1, r1, r12, lsl #2
; CHECK-NEXT:    add.w r2, r2, r12, lsl #2
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB2_7: @ %for.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldr s0, [r0]
; CHECK-NEXT:    adds r0, #4
; CHECK-NEXT:    vldr s2, [r1]
; CHECK-NEXT:    adds r1, #4
; CHECK-NEXT:    vldr s4, [r2]
; CHECK-NEXT:    vfma.f32 s4, s2, s0
; CHECK-NEXT:    vstr s4, [r2]
; CHECK-NEXT:    adds r2, #4
; CHECK-NEXT:    le lr, .LBB2_7
; CHECK-NEXT:  .LBB2_8: @ %for.cond.cleanup
; CHECK-NEXT:    pop {r4, r5, r6, pc}
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %min.iters.check = icmp ult i32 %n, 8
  br i1 %min.iters.check, label %for.body.preheader12, label %vector.ph

for.body.preheader12:                             ; preds = %middle.block, %for.body.preheader
  %i.09.ph = phi i32 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds float, float* %A, i32 %index
  %1 = bitcast float* %0 to <8 x float>*
  %wide.load = load <8 x float>, <8 x float>* %1, align 4
  %2 = getelementptr inbounds float, float* %B, i32 %index
  %3 = bitcast float* %2 to <8 x float>*
  %wide.load10 = load <8 x float>, <8 x float>* %3, align 4
  %4 = fmul fast <8 x float> %wide.load10, %wide.load
  %5 = getelementptr inbounds float, float* %C, i32 %index
  %6 = bitcast float* %5 to <8 x float>*
  %wide.load11 = load <8 x float>, <8 x float>* %6, align 4
  %7 = fadd fast <8 x float> %wide.load11, %4
  store <8 x float> %7, <8 x float>* %6, align 4
  %index.next = add i32 %index, 8
  %8 = icmp eq i32 %index.next, %n.vec
  br i1 %8, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %n.vec, %n
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader12

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader12, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ %i.09.ph, %for.body.preheader12 ]
  %arrayidx = getelementptr inbounds float, float* %A, i32 %i.09
  %9 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %B, i32 %i.09
  %10 = load float, float* %arrayidx1, align 4
  %mul = fmul fast float %10, %9
  %arrayidx2 = getelementptr inbounds float, float* %C, i32 %i.09
  %11 = load float, float* %arrayidx2, align 4
  %add = fadd fast float %11, %mul
  store float %add, float* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare i32 @llvm.arm.mve.addv.v4i32(<4 x i32>, i32)
declare { i32, i32 } @llvm.arm.mve.vmlldava.v8i16(i32, i32, i32, i32, i32, <8 x i16>, <8 x i16>)
