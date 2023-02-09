; ModuleID = './llvm/test/Linker/Inputs/old_global_ctors.3.4.bc'
source_filename = "./llvm/test/Linker/Inputs/old_global_ctors.3.4.bc"

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @a_global_ctor, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @b_global_ctor, i8* null }]

declare void @a_global_ctor()

declare void @b_global_ctor()
