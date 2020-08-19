; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -S -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -loop-unroll | FileCheck %s
; RUN: opt < %s -S -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -loop-unroll | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: norecurse nounwind
define i8* @f(i8* returned %s, i32 zeroext %x, i32 signext %k) local_unnamed_addr #0 {
; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP10:%.*]] = icmp sgt i32 [[K:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP10]], label [[FOR_BODY_LR_PH:%.*]], label [[FOR_END:%.*]]
; CHECK:       for.body.lr.ph:
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[K]] to i64
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i32 [[K]], 16
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[FOR_BODY_PREHEADER:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i64 [[WIDE_TRIP_COUNT]], 4294967280
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <16 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <16 x i32> [[BROADCAST_SPLATINSERT]], <16 x i32> undef, <16 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP0:%.*]] = add nsw i64 [[N_VEC]], -16
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; CHECK-NEXT:    [[XTRAITER1:%.*]] = and i64 [[TMP2]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP1]], 1
; CHECK-NEXT:    br i1 [[TMP3]], label [[MIDDLE_BLOCK_UNR_LCSSA:%.*]], label [[VECTOR_PH_NEW:%.*]]
; CHECK:       vector.ph.new:
; CHECK-NEXT:    [[UNROLL_ITER:%.*]] = sub i64 [[TMP2]], [[XTRAITER1]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH_NEW]] ], [ [[INDEX_NEXT_1:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND12:%.*]] = phi <16 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, [[VECTOR_PH_NEW]] ], [ [[VEC_IND_NEXT13_1:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[NITER:%.*]] = phi i64 [ [[UNROLL_ITER]], [[VECTOR_PH_NEW]] ], [ [[NITER_NSUB_1:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = shl <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, [[VEC_IND12]]
; CHECK-NEXT:    [[TMP5:%.*]] = and <16 x i32> [[TMP4]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq <16 x i32> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = select <16 x i1> [[TMP6]], <16 x i8> <i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48>, <16 x i8> <i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49>
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i8, i8* [[S:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i8* [[TMP8]] to <16 x i8>*
; CHECK-NEXT:    store <16 x i8> [[TMP7]], <16 x i8>* [[TMP9]], align 1
; CHECK-NEXT:    [[INDEX_NEXT:%.*]] = add nuw nsw i64 [[INDEX]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT13:%.*]] = add <16 x i32> [[VEC_IND12]], <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; CHECK-NEXT:    [[NITER_NSUB:%.*]] = sub i64 [[NITER]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = shl <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, [[VEC_IND_NEXT13]]
; CHECK-NEXT:    [[TMP11:%.*]] = and <16 x i32> [[TMP10]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq <16 x i32> [[TMP11]], zeroinitializer
; CHECK-NEXT:    [[TMP13:%.*]] = select <16 x i1> [[TMP12]], <16 x i8> <i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48>, <16 x i8> <i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49>
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDEX_NEXT]]
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast i8* [[TMP14]] to <16 x i8>*
; CHECK-NEXT:    store <16 x i8> [[TMP13]], <16 x i8>* [[TMP15]], align 1
; CHECK-NEXT:    [[INDEX_NEXT_1]] = add i64 [[INDEX_NEXT]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT13_1]] = add <16 x i32> [[VEC_IND_NEXT13]], <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; CHECK-NEXT:    [[NITER_NSUB_1]] = sub i64 [[NITER_NSUB]], 1
; CHECK-NEXT:    [[NITER_NCMP_1:%.*]] = icmp eq i64 [[NITER_NSUB_1]], 0
; CHECK-NEXT:    br i1 [[NITER_NCMP_1]], label [[MIDDLE_BLOCK_UNR_LCSSA_LOOPEXIT:%.*]], label [[VECTOR_BODY]]
; CHECK:       middle.block.unr-lcssa.loopexit:
; CHECK-NEXT:    [[INDEX_UNR_PH:%.*]] = phi i64 [ [[INDEX_NEXT_1]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND12_UNR_PH:%.*]] = phi <16 x i32> [ [[VEC_IND_NEXT13_1]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    br label [[MIDDLE_BLOCK_UNR_LCSSA]]
; CHECK:       middle.block.unr-lcssa:
; CHECK-NEXT:    [[INDEX_UNR:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_UNR_PH]], [[MIDDLE_BLOCK_UNR_LCSSA_LOOPEXIT]] ]
; CHECK-NEXT:    [[VEC_IND12_UNR:%.*]] = phi <16 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, [[VECTOR_PH]] ], [ [[VEC_IND12_UNR_PH]], [[MIDDLE_BLOCK_UNR_LCSSA_LOOPEXIT]] ]
; CHECK-NEXT:    [[LCMP_MOD2:%.*]] = icmp ne i64 [[XTRAITER1]], 0
; CHECK-NEXT:    br i1 [[LCMP_MOD2]], label [[VECTOR_BODY_EPIL_PREHEADER:%.*]], label [[MIDDLE_BLOCK:%.*]]
; CHECK:       vector.body.epil.preheader:
; CHECK-NEXT:    br label [[VECTOR_BODY_EPIL:%.*]]
; CHECK:       vector.body.epil:
; CHECK-NEXT:    [[INDEX_EPIL:%.*]] = phi i64 [ [[INDEX_UNR]], [[VECTOR_BODY_EPIL_PREHEADER]] ]
; CHECK-NEXT:    [[VEC_IND12_EPIL:%.*]] = phi <16 x i32> [ [[VEC_IND12_UNR]], [[VECTOR_BODY_EPIL_PREHEADER]] ]
; CHECK-NEXT:    [[TMP16:%.*]] = shl <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, [[VEC_IND12_EPIL]]
; CHECK-NEXT:    [[TMP17:%.*]] = and <16 x i32> [[TMP16]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP18:%.*]] = icmp eq <16 x i32> [[TMP17]], zeroinitializer
; CHECK-NEXT:    [[TMP19:%.*]] = select <16 x i1> [[TMP18]], <16 x i8> <i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48>, <16 x i8> <i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49>
; CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDEX_EPIL]]
; CHECK-NEXT:    [[TMP21:%.*]] = bitcast i8* [[TMP20]] to <16 x i8>*
; CHECK-NEXT:    store <16 x i8> [[TMP19]], <16 x i8>* [[TMP21]], align 1
; CHECK-NEXT:    [[INDEX_NEXT_EPIL:%.*]] = add i64 [[INDEX_EPIL]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT13_EPIL:%.*]] = add <16 x i32> [[VEC_IND12_EPIL]], <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; CHECK-NEXT:    [[TMP22:%.*]] = icmp eq i64 [[INDEX_NEXT_EPIL]], [[N_VEC]]
; CHECK-NEXT:    br label [[MIDDLE_BLOCK_EPILOG_LCSSA:%.*]]
; CHECK:       middle.block.epilog-lcssa:
; CHECK-NEXT:    br label [[MIDDLE_BLOCK]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_VEC]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END]], label [[FOR_BODY_PREHEADER]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[INDVARS_IV_PH:%.*]] = phi i64 [ 0, [[FOR_BODY_LR_PH]] ], [ [[N_VEC]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    [[TMP23:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[INDVARS_IV_PH]]
; CHECK-NEXT:    [[TMP24:%.*]] = add i64 [[WIDE_TRIP_COUNT]], -1
; CHECK-NEXT:    [[TMP25:%.*]] = sub i64 [[TMP24]], [[INDVARS_IV_PH]]
; CHECK-NEXT:    [[XTRAITER:%.*]] = and i64 [[TMP23]], 7
; CHECK-NEXT:    [[LCMP_MOD:%.*]] = icmp ne i64 [[XTRAITER]], 0
; CHECK-NEXT:    br i1 [[LCMP_MOD]], label [[FOR_BODY_PROL_PREHEADER:%.*]], label [[FOR_BODY_PROL_LOOPEXIT:%.*]]
; CHECK:       for.body.prol.preheader:
; CHECK-NEXT:    br label [[FOR_BODY_PROL:%.*]]
; CHECK:       for.body.prol:
; CHECK-NEXT:    [[INDVARS_IV_PROL:%.*]] = phi i64 [ [[INDVARS_IV_NEXT_PROL:%.*]], [[FOR_BODY_PROL]] ], [ [[INDVARS_IV_PH]], [[FOR_BODY_PROL_PREHEADER]] ]
; CHECK-NEXT:    [[PROL_ITER:%.*]] = phi i64 [ [[XTRAITER]], [[FOR_BODY_PROL_PREHEADER]] ], [ [[PROL_ITER_SUB:%.*]], [[FOR_BODY_PROL]] ]
; CHECK-NEXT:    [[TMP26:%.*]] = trunc i64 [[INDVARS_IV_PROL]] to i32
; CHECK-NEXT:    [[SHL_PROL:%.*]] = shl i32 1, [[TMP26]]
; CHECK-NEXT:    [[AND_PROL:%.*]] = and i32 [[SHL_PROL]], [[X]]
; CHECK-NEXT:    [[TOBOOL_PROL:%.*]] = icmp eq i32 [[AND_PROL]], 0
; CHECK-NEXT:    [[CONV_PROL:%.*]] = select i1 [[TOBOOL_PROL]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_PROL:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_PROL]]
; CHECK-NEXT:    store i8 [[CONV_PROL]], i8* [[ARRAYIDX_PROL]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_PROL]] = add nuw nsw i64 [[INDVARS_IV_PROL]], 1
; CHECK-NEXT:    [[EXITCOND_PROL:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT_PROL]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    [[PROL_ITER_SUB]] = sub i64 [[PROL_ITER]], 1
; CHECK-NEXT:    [[PROL_ITER_CMP:%.*]] = icmp ne i64 [[PROL_ITER_SUB]], 0
; CHECK-NEXT:    br i1 [[PROL_ITER_CMP]], label [[FOR_BODY_PROL]], label [[FOR_BODY_PROL_LOOPEXIT_UNR_LCSSA:%.*]], !llvm.loop !0
; CHECK:       for.body.prol.loopexit.unr-lcssa:
; CHECK-NEXT:    [[INDVARS_IV_UNR_PH:%.*]] = phi i64 [ [[INDVARS_IV_NEXT_PROL]], [[FOR_BODY_PROL]] ]
; CHECK-NEXT:    br label [[FOR_BODY_PROL_LOOPEXIT]]
; CHECK:       for.body.prol.loopexit:
; CHECK-NEXT:    [[INDVARS_IV_UNR:%.*]] = phi i64 [ [[INDVARS_IV_PH]], [[FOR_BODY_PREHEADER]] ], [ [[INDVARS_IV_UNR_PH]], [[FOR_BODY_PROL_LOOPEXIT_UNR_LCSSA]] ]
; CHECK-NEXT:    [[TMP27:%.*]] = icmp ult i64 [[TMP25]], 7
; CHECK-NEXT:    br i1 [[TMP27]], label [[FOR_END_LOOPEXIT:%.*]], label [[FOR_BODY_PREHEADER_NEW:%.*]]
; CHECK:       for.body.preheader.new:
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_UNR]], [[FOR_BODY_PREHEADER_NEW]] ], [ [[INDVARS_IV_NEXT_7:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[TMP28:%.*]] = trunc i64 [[INDVARS_IV]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 1, [[TMP28]]
; CHECK-NEXT:    [[AND:%.*]] = and i32 [[SHL]], [[X]]
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[AND]], 0
; CHECK-NEXT:    [[CONV:%.*]] = select i1 [[TOBOOL]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    store i8 [[CONV]], i8* [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT:%.*]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[TMP29:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; CHECK-NEXT:    [[SHL_1:%.*]] = shl i32 1, [[TMP29]]
; CHECK-NEXT:    [[AND_1:%.*]] = and i32 [[SHL_1]], [[X]]
; CHECK-NEXT:    [[TOBOOL_1:%.*]] = icmp eq i32 [[AND_1]], 0
; CHECK-NEXT:    [[CONV_1:%.*]] = select i1 [[TOBOOL_1]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_1:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT]]
; CHECK-NEXT:    store i8 [[CONV_1]], i8* [[ARRAYIDX_1]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_1:%.*]] = add nuw nsw i64 [[INDVARS_IV_NEXT]], 1
; CHECK-NEXT:    [[TMP30:%.*]] = trunc i64 [[INDVARS_IV_NEXT_1]] to i32
; CHECK-NEXT:    [[SHL_2:%.*]] = shl i32 1, [[TMP30]]
; CHECK-NEXT:    [[AND_2:%.*]] = and i32 [[SHL_2]], [[X]]
; CHECK-NEXT:    [[TOBOOL_2:%.*]] = icmp eq i32 [[AND_2]], 0
; CHECK-NEXT:    [[CONV_2:%.*]] = select i1 [[TOBOOL_2]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_2:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT_1]]
; CHECK-NEXT:    store i8 [[CONV_2]], i8* [[ARRAYIDX_2]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_2:%.*]] = add nuw nsw i64 [[INDVARS_IV_NEXT_1]], 1
; CHECK-NEXT:    [[TMP31:%.*]] = trunc i64 [[INDVARS_IV_NEXT_2]] to i32
; CHECK-NEXT:    [[SHL_3:%.*]] = shl i32 1, [[TMP31]]
; CHECK-NEXT:    [[AND_3:%.*]] = and i32 [[SHL_3]], [[X]]
; CHECK-NEXT:    [[TOBOOL_3:%.*]] = icmp eq i32 [[AND_3]], 0
; CHECK-NEXT:    [[CONV_3:%.*]] = select i1 [[TOBOOL_3]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_3:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT_2]]
; CHECK-NEXT:    store i8 [[CONV_3]], i8* [[ARRAYIDX_3]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_3:%.*]] = add nuw nsw i64 [[INDVARS_IV_NEXT_2]], 1
; CHECK-NEXT:    [[TMP32:%.*]] = trunc i64 [[INDVARS_IV_NEXT_3]] to i32
; CHECK-NEXT:    [[SHL_4:%.*]] = shl i32 1, [[TMP32]]
; CHECK-NEXT:    [[AND_4:%.*]] = and i32 [[SHL_4]], [[X]]
; CHECK-NEXT:    [[TOBOOL_4:%.*]] = icmp eq i32 [[AND_4]], 0
; CHECK-NEXT:    [[CONV_4:%.*]] = select i1 [[TOBOOL_4]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_4:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT_3]]
; CHECK-NEXT:    store i8 [[CONV_4]], i8* [[ARRAYIDX_4]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_4:%.*]] = add nuw nsw i64 [[INDVARS_IV_NEXT_3]], 1
; CHECK-NEXT:    [[TMP33:%.*]] = trunc i64 [[INDVARS_IV_NEXT_4]] to i32
; CHECK-NEXT:    [[SHL_5:%.*]] = shl i32 1, [[TMP33]]
; CHECK-NEXT:    [[AND_5:%.*]] = and i32 [[SHL_5]], [[X]]
; CHECK-NEXT:    [[TOBOOL_5:%.*]] = icmp eq i32 [[AND_5]], 0
; CHECK-NEXT:    [[CONV_5:%.*]] = select i1 [[TOBOOL_5]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_5:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT_4]]
; CHECK-NEXT:    store i8 [[CONV_5]], i8* [[ARRAYIDX_5]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_5:%.*]] = add nuw nsw i64 [[INDVARS_IV_NEXT_4]], 1
; CHECK-NEXT:    [[TMP34:%.*]] = trunc i64 [[INDVARS_IV_NEXT_5]] to i32
; CHECK-NEXT:    [[SHL_6:%.*]] = shl i32 1, [[TMP34]]
; CHECK-NEXT:    [[AND_6:%.*]] = and i32 [[SHL_6]], [[X]]
; CHECK-NEXT:    [[TOBOOL_6:%.*]] = icmp eq i32 [[AND_6]], 0
; CHECK-NEXT:    [[CONV_6:%.*]] = select i1 [[TOBOOL_6]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_6:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT_5]]
; CHECK-NEXT:    store i8 [[CONV_6]], i8* [[ARRAYIDX_6]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_6:%.*]] = add nuw nsw i64 [[INDVARS_IV_NEXT_5]], 1
; CHECK-NEXT:    [[TMP35:%.*]] = trunc i64 [[INDVARS_IV_NEXT_6]] to i32
; CHECK-NEXT:    [[SHL_7:%.*]] = shl i32 1, [[TMP35]]
; CHECK-NEXT:    [[AND_7:%.*]] = and i32 [[SHL_7]], [[X]]
; CHECK-NEXT:    [[TOBOOL_7:%.*]] = icmp eq i32 [[AND_7]], 0
; CHECK-NEXT:    [[CONV_7:%.*]] = select i1 [[TOBOOL_7]], i8 48, i8 49
; CHECK-NEXT:    [[ARRAYIDX_7:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[INDVARS_IV_NEXT_6]]
; CHECK-NEXT:    store i8 [[CONV_7]], i8* [[ARRAYIDX_7]], align 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT_7]] = add nuw nsw i64 [[INDVARS_IV_NEXT_6]], 1
; CHECK-NEXT:    [[EXITCOND_7:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT_7]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[EXITCOND_7]], label [[FOR_END_LOOPEXIT_UNR_LCSSA:%.*]], label [[FOR_BODY]]
; CHECK:       for.end.loopexit.unr-lcssa:
; CHECK-NEXT:    br label [[FOR_END_LOOPEXIT]]
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    [[IDXPROM1:%.*]] = sext i32 [[K]] to i64
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i8, i8* [[S]], i64 [[IDXPROM1]]
; CHECK-NEXT:    store i8 0, i8* [[ARRAYIDX2]], align 1
; CHECK-NEXT:    ret i8* [[S]]
;
entry:
  %cmp10 = icmp sgt i32 %k, 0
  br i1 %cmp10, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %wide.trip.count = zext i32 %k to i64
  %min.iters.check = icmp ult i32 %k, 16
  br i1 %min.iters.check, label %for.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i64 %wide.trip.count, 4294967280
  %broadcast.splatinsert = insertelement <16 x i32> undef, i32 %x, i32 0
  %broadcast.splat = shufflevector <16 x i32> %broadcast.splatinsert, <16 x i32> undef, <16 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind12 = phi <16 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, %vector.ph ], [ %vec.ind.next13, %vector.body ]
  %0 = shl <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, %vec.ind12
  %1 = and <16 x i32> %0, %broadcast.splat
  %2 = icmp eq <16 x i32> %1, zeroinitializer
  %3 = select <16 x i1> %2, <16 x i8> <i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48>, <16 x i8> <i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49>
  %4 = getelementptr inbounds i8, i8* %s, i64 %index
  %5 = bitcast i8* %4 to <16 x i8>*
  store <16 x i8> %3, <16 x i8>* %5, align 1
  %index.next = add i64 %index, 16
  %vec.ind.next13 = add <16 x i32> %vec.ind12, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %6 = icmp eq i64 %index.next, %n.vec
  br i1 %6, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %middle.block, %for.body.lr.ph
  %indvars.iv.ph = phi i64 [ 0, %for.body.lr.ph ], [ %n.vec, %middle.block ]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader ]
  %7 = trunc i64 %indvars.iv to i32
  %shl = shl i32 1, %7
  %and = and i32 %shl, %x
  %tobool = icmp eq i32 %and, 0
  %conv = select i1 %tobool, i8 48, i8 49
  %arrayidx = getelementptr inbounds i8, i8* %s, i64 %indvars.iv
  store i8 %conv, i8* %arrayidx, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %middle.block, %entry
  %idxprom1 = sext i32 %k to i64
  %arrayidx2 = getelementptr inbounds i8, i8* %s, i64 %idxprom1
  store i8 0, i8* %arrayidx2, align 1
  ret i8* %s
}

