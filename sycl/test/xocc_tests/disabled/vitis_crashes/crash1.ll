; ModuleID = 'reduced.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64_sw_emu-xilinx-unknown-sycldevice"

%"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec" = type { <3 x double> }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }

define weak_odr dso_local spir_kernel void @el_800_480_100_5_487_WRWNjd720(%"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec" addrspace(1)* %_arg_m_frame_ptr, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_m_frame_ptr3) local_unnamed_addr !kernel_arg_buffer_location !0 {
label_0:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_m_frame_ptr3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %2 = tail call i64 @llvm.spir.get.global.id.i64(i32 0)
  br label %label_1

label_1:                                          ; preds = %label_4, %label_0
  %i.0.i = phi i32 [ 0, %label_0 ], [ %inc.i, %label_4 ]
  %cmp.i = icmp ult i32 %i.0.i, 100
  br i1 %cmp.i, label %label_2, label %label_5

label_2:                                          ; preds = %label_2, %label_1
  %call3.i.i.i.i.i.i = call spir_func double @_Z3fmaddd()
  %cmp.i.i.i = fcmp ult double %call3.i.i.i.i.i.i, 1.000000e+00
  br i1 %cmp.i.i.i, label %label_3, label %label_2

label_3:                                          ; preds = %label_2
  br label %label_4

label_4:                                          ; preds = %label_3
  %inc.i = add nuw nsw i32 %i.0.i, 1
  br label %label_1

label_5:                                          ; preds = %label_1
  %add.ptr.i = getelementptr inbounds %"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec", %"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec" addrspace(1)* %_arg_m_frame_ptr, i64 %1
  %mul.i = mul i64 undef, 800
  %add.i = add i64 %mul.i, %2
  %ptridx.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec", %"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec" addrspace(1)* %add.ptr.i, i64 %add.i
  %3 = bitcast %"class._ZTSN2cl4sycl3vecIdLi3EEE.cl::sycl::vec" addrspace(1)* %ptridx.i.i to <4 x double> addrspace(1)*
  store <4 x double> undef, <4 x double> addrspace(1)* %3, align 32, !tbaa.struct !1
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind readnone
declare dso_local spir_func double @_Z3fmaddd() local_unnamed_addr #1

; Function Attrs: nounwind
declare void @llvm.assume(i1) #2

declare i64 @llvm.spir.get.global.id.i64(i32)

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!1 = !{i64 0, i64 32, !2}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
