; ModuleID = './llvm/test/Bitcode/arm-intrinsics.bc'
source_filename = "llvm/test/Bitcode/arm-intrinsics.ll"

define void @f(i32* %p) {
  %a = call i32 @llvm.arm.ldrex.p0i32(i32* %p)
  %c = call i32 @llvm.arm.strex.p0i32(i32 0, i32* %p)
  %a2 = call i32 @llvm.arm.ldaex.p0i32(i32* %p)
  %c2 = call i32 @llvm.arm.stlex.p0i32(i32 0, i32* %p)
  ret void
}

; Function Attrs: nounwind
declare i32 @llvm.arm.ldrex.p0i32(i32*) #0

; Function Attrs: nounwind
declare i32 @llvm.arm.ldaex.p0i32(i32*) #0

; Function Attrs: nounwind
declare i32 @llvm.arm.stlex.p0i32(i32, i32*) #0

; Function Attrs: nounwind
declare i32 @llvm.arm.strex.p0i32(i32, i32*) #0

attributes #0 = { nounwind }
