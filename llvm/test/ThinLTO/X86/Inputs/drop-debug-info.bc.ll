; ModuleID = './llvm/test/ThinLTO/X86/Inputs/drop-debug-info.bc'
source_filename = "/Users/amini/projects/vanilla/llvm/test/ThinLTO/X86/Inputs/drop-debug-info.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@argc = global i8 0, align 1

define void @globalfunc() {
entry:
  %0 = load i8, i8* @argc, align 1
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 0}
