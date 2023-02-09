; ModuleID = './llvm/test/Bitcode/calling-conventions.3.2.ll.bc'
source_filename = ".\\llvm\\test\\Bitcode\\calling-conventions.3.2.ll"

declare void @ccc()

declare fastcc void @fastcc()

declare coldcc void @coldcc()

declare ghccc void @cc10()

declare spir_kernel void @spir_kernel()

declare spir_func void @spir_func()

declare intel_ocl_bicc void @intel_ocl_bicc()

declare cfguard_checkcc void @cfguard_checkcc()

declare x86_stdcallcc void @x86_stdcallcc()

declare x86_fastcallcc void @x86_fastcallcc()

declare x86_thiscallcc void @x86_thiscallcc()

declare arm_apcscc void @arm_apcscc()

declare arm_aapcscc void @arm_aapcscc()

declare arm_aapcs_vfpcc void @arm_aapcs_vfpcc()

declare msp430_intrcc void @msp430_intrcc()

declare ptx_kernel void @ptx_kernel()

declare ptx_device void @ptx_device()

define void @call_ccc() {
  call void @ccc()
  ret void
}

define void @call_fastcc() {
  call fastcc void @fastcc()
  ret void
}

define void @call_coldcc() {
  call coldcc void @coldcc()
  ret void
}

define void @call_cc10() {
  call ghccc void @cc10()
  ret void
}

define void @call_spir_kernel() {
  call spir_kernel void @spir_kernel()
  ret void
}

define void @call_spir_func() {
  call spir_func void @spir_func()
  ret void
}

define void @call_intel_ocl_bicc() {
  call intel_ocl_bicc void @intel_ocl_bicc()
  ret void
}

define void @call_cfguard_checkcc() {
  call cfguard_checkcc void @cfguard_checkcc()
  ret void
}

define void @call_x86_stdcallcc() {
  call x86_stdcallcc void @x86_stdcallcc()
  ret void
}

define void @call_x86_fastcallcc() {
  call x86_fastcallcc void @x86_fastcallcc()
  ret void
}

define void @call_x86_thiscallcc() {
  call x86_thiscallcc void @x86_thiscallcc()
  ret void
}

define void @call_arm_apcscc() {
  call arm_apcscc void @arm_apcscc()
  ret void
}

define void @call_arm_aapcscc() {
  call arm_aapcscc void @arm_aapcscc()
  ret void
}

define void @call_arm_aapcs_vfpcc() {
  call arm_aapcs_vfpcc void @arm_aapcs_vfpcc()
  ret void
}

define void @call_msp430_intrcc() {
  call msp430_intrcc void @msp430_intrcc()
  ret void
}

define void @call_ptx_kernel() {
  call ptx_kernel void @ptx_kernel()
  ret void
}

define void @call_ptx_device() {
  call ptx_device void @ptx_device()
  ret void
}
