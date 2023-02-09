; ModuleID = './llvm/test/Bitcode/conversionInstructions.3.2.ll.bc'
source_filename = "./llvm/test/Bitcode/conversionInstructions.3.2.ll.bc"

define void @trunc(i32 %src) {
entry:
  %res1 = trunc i32 %src to i8
  ret void
}

define void @zext(i32 %src) {
entry:
  %res1 = zext i32 %src to i64
  ret void
}

define void @sext(i32 %src) {
entry:
  %res1 = sext i32 %src to i64
  ret void
}

define void @fptrunc(double %src) {
entry:
  %res1 = fptrunc double %src to float
  ret void
}

define void @fpext(float %src) {
entry:
  %res1 = fpext float %src to double
  ret void
}

define void @fptoui(float %src) {
entry:
  %res1 = fptoui float %src to i32
  ret void
}

define void @fptosi(float %src) {
entry:
  %res1 = fptosi float %src to i32
  ret void
}

define void @uitofp(i32 %src) {
entry:
  %res1 = uitofp i32 %src to float
  ret void
}

define void @sitofp(i32 %src) {
entry:
  %res1 = sitofp i32 %src to float
  ret void
}

define void @ptrtoint(i32* %src) {
entry:
  %res1 = ptrtoint i32* %src to i8
  ret void
}

define void @inttoptr(i32 %src) {
entry:
  %res1 = inttoptr i32 %src to i32*
  ret void
}

define void @bitcast(i32 %src1, i32* %src2) {
entry:
  %res1 = bitcast i32 %src1 to i32
  %res2 = bitcast i32* %src2 to i64*
  ret void
}

define void @ptrtointInstr(i32* %ptr, <4 x i32*> %vecPtr) {
entry:
  %res1 = ptrtoint i32* %ptr to i8
  %res2 = ptrtoint <4 x i32*> %vecPtr to <4 x i64>
  ret void
}

define void @inttoptrInstr(i32 %x, <4 x i32> %vec) {
entry:
  %res1 = inttoptr i32 %x to i64*
  %res2 = inttoptr <4 x i32> %vec to <4 x i8*>
  ret void
}
