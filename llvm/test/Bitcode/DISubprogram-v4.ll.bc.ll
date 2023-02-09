; ModuleID = './llvm/test/Bitcode/DISubprogram-v4.ll.bc'
source_filename = "<stdin>"

define void @_Z3foov() !dbg !9 {
  ret void
}

!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18}
!llvm.module.flags = !{!19}
!llvm.dbg.cu = !{!8}

!0 = !{null}
!1 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DISubroutineType(types: !0)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type)
!5 = distinct !{}
!6 = distinct !{}
!7 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !8)
!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: true, flags: "-O2", runtimeVersion: 0, emissionKind: NoDebug)
!9 = distinct !DISubprogram(scope: null, spFlags: 0)
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, virtualIndex: 0, spFlags: DISPFlagPureVirtual | DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8)
!11 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, virtualIndex: 0, spFlags: DISPFlagVirtual | DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8)
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8)
!13 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8)
!14 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized)
!15 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !8)
!16 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagLocalToUnit)
!17 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagDefinition, unit: !8)
!18 = distinct !DISubprogram(name: "foo", linkageName: "_Zfoov", scope: !1, file: !2, line: 7, type: !3, containingType: !4, spFlags: DISPFlagOptimized)
!19 = !{i32 1, !"Debug Info Version", i32 3}
