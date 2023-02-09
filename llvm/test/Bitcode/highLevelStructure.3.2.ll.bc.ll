; ModuleID = './llvm/test/Bitcode/highLevelStructure.3.2.ll.bc'
source_filename = "./llvm/test/Bitcode/highLevelStructure.3.2.ll.bc"
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-a0:0:64-f80:32:32-n8:16:32-S32"

module asm "some assembly"

%mytype = type { %mytype*, i32 }

@glob1 = global i32 1

@aliased1 = alias i32, i32* @glob1
@aliased2 = internal alias i32, i32* @glob1
@aliased3 = alias i32, i32* @glob1
@aliased4 = weak alias i32, i32* @glob1
@aliased5 = weak_odr alias i32, i32* @glob1

declare void @ParamAttr1(i8 zeroext)

declare void @ParamAttr2(i8* nest)

declare void @ParamAttr3(i8* sret(i8))

declare void @ParamAttr4(i8 signext)

declare void @ParamAttr5(i8* inreg)

declare void @ParamAttr6(i8* byval(i8))

declare void @ParamAttr7(i8* noalias)

declare void @ParamAttr8(i8* nocapture)

declare void @ParamAttr9(i8* nest noalias nocapture)

declare void @ParamAttr10(i8* noalias nocapture sret(i8))

declare void @ParamAttr11(i8* noalias nocapture byval(i8))

declare void @ParamAttr12(i8* inreg noalias nocapture)

define void @NamedTypes() {
entry:
  %res = alloca %mytype, align 8
  ret void
}

define void @gcTest() gc "gc" {
entry:
  ret void
}

!name = !{!0, !1, !2}

!0 = !{!"zero"}
!1 = !{!"one"}
!2 = !{!"two"}
