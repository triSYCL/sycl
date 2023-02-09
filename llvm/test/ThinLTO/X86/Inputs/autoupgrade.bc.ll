; ModuleID = './llvm/test/ThinLTO/X86/Inputs/autoupgrade.bc'
source_filename = "funcimport.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc1() {
entry:
  %tmp = call {}* @llvm.invariant.start.p0i8(i64 4, i8* null)
  ret void
}

define void @globalfunc2() {
entry:
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare {}* @llvm.invariant.start.p0i8(i64 immarg, i8* nocapture) #0

attributes #0 = { argmemonly nofree nosync nounwind willreturn }
