; ModuleID = './llvm/test/Bitcode/global-variables.3.2.ll.bc'
source_filename = "./llvm/test/Bitcode/global-variables.3.2.ll.bc"

@global.var = global i32 1
@constant.var = constant i32 1
@noinit.var = global float undef
@section.var = global i32 1, section "foo"
@align.var = global i64 undef, align 8
@unnamed_addr.var = unnamed_addr global i8 1
@default_addrspace.var = global i8 1
@non_default_addrspace.var = addrspace(1) global i8* undef
@initialexec.var = thread_local(initialexec) global i32 0, align 4
@localdynamic.var = thread_local(localdynamic) constant i32 0, align 4
@localexec.var = thread_local(localexec) constant i32 0, align 4
@string.var = private unnamed_addr constant [13 x i8] c"hello world\0A\00"
