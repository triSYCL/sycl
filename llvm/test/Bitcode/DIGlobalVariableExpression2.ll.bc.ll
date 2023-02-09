; ModuleID = './llvm/test/Bitcode/DIGlobalVariableExpression2.ll.bc'
source_filename = "./llvm/test/Bitcode/DIGlobalVariableExpression2.ll.bc"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@g = common global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.7.0 (clang-stage1-configure-RA_build 241111)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "a.c", directory: "/tmp")
!4 = !{}
!5 = !{!6}
!6 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version 3.7.0 (clang-stage1-configure-RA_build 241111)"}
