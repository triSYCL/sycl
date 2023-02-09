; ModuleID = './llvm/test/Bitcode/upgrade-var-annotation.ll.bc'
source_filename = "upgrade-var-annotation.ll"

define void @f(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3) {
  call void @llvm.var.annotation(i8* %arg0, i8* %arg1, i8* %arg2, i32 %arg3, i8* null)
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #0

attributes #0 = { inaccessiblememonly nofree nosync nounwind willreturn }
