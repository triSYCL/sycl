; ModuleID = './llvm/test/Bitcode/thinlto-summary-local-5.0.ll.bc'
source_filename = "thinlto-summary-local-5.0.ll"

@bar = global i32 0

@baz = alias i32, i32* @bar

define void @foo() {
  ret void
}
