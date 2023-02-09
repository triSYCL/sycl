; ModuleID = './llvm/test/Bitcode/upgrade-objcretainrelease-asm.ll.bc'
source_filename = "upgrade-objcretainrelease-asm.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

define void @inlineasm() {
  call void asm sideeffect "mov\09fp, fp\09\09; marker for objc_retainAutoreleaseReturnValue", ""()
  ret void
}
