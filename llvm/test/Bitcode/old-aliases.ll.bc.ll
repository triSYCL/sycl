; ModuleID = './llvm/test/Bitcode/old-aliases.ll.bc'
source_filename = "./llvm/test/Bitcode/old-aliases.ll.bc"

@v1 = global i32 0
@v2 = global [1 x i32] zeroinitializer

@v3 = alias i16, bitcast (i32* @v1 to i16*)
@v4 = alias i32, getelementptr inbounds ([1 x i32], [1 x i32]* @v2, i32 0, i32 0)
@v5 = alias i32, addrspacecast (i32* @v1 to i32 addrspace(2)*)
@v6 = alias i16, i16* @v3
