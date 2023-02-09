; ModuleID = './llvm/test/Bitcode/upgrade-aarch64-ldstxr.bc'
source_filename = "/tmp/a.ll"

define void @f(i32* %p) {
  %a = call i64 @llvm.aarch64.ldxr.p0i32(i32* %p)
  %c = call i32 @llvm.aarch64.stxr.p0i32(i64 0, i32* %p)
  %a2 = call i64 @llvm.aarch64.ldaxr.p0i32(i32* %p)
  %c2 = call i32 @llvm.aarch64.stlxr.p0i32(i64 0, i32* %p)
  ret void
}

; Function Attrs: nofree nounwind willreturn
declare i64 @llvm.aarch64.ldxr.p0i32(i32*) #0

; Function Attrs: nofree nounwind willreturn
declare i64 @llvm.aarch64.ldaxr.p0i32(i32*) #0

; Function Attrs: nofree nounwind willreturn
declare i32 @llvm.aarch64.stxr.p0i32(i64, i32*) #0

; Function Attrs: nofree nounwind willreturn
declare i32 @llvm.aarch64.stlxr.p0i32(i64, i32*) #0

attributes #0 = { nofree nounwind willreturn }
