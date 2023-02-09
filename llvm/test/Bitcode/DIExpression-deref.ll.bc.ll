; ModuleID = './llvm/test/Bitcode/DIExpression-deref.ll.bc'
source_filename = "/Volumes/Fusion/Data/llvm/test/Bitcode/DIExpression-deref.ll"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang (llvm/trunk 300520)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{}
!3 = !{!4, !7, !8, !9}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_plus_uconst, 0, DW_OP_deref, DW_OP_LLVM_fragment, 8, 8))
!5 = distinct !DIGlobalVariable(name: "g", scope: !0, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_plus_uconst, 0, DW_OP_deref))
!8 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_plus_uconst, 1, DW_OP_deref))
!9 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_deref))
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
