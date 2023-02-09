; ModuleID = './llvm/test/Bitcode/constantsTest.3.2.ll.bc'
source_filename = "./llvm/test/Bitcode/constantsTest.3.2.ll.bc"

@X = global i32 0
@Y = global i32 1
@Z = global [2 x i32*] [i32* @X, i32* @Y]

define void @SimpleConstants(i32 %x) {
entry:
  store i32 %x, i32* null, align 4
  %res1 = fcmp true float 1.000000e+00, 1.000000e+00
  %res2 = fcmp false float 1.000000e+00, 1.000000e+00
  %res3 = add i32 0, 0
  %res4 = fadd float 0.000000e+00, 0.000000e+00
  ret void
}

define void @ComplexConstants(<2 x i32> %x) {
entry:
  %res1 = extractvalue { i32, float } { i32 1, float 2.000000e+00 }, 0
  %res2 = extractvalue [2 x i32] [i32 1, i32 2], 0
  %res3 = add <2 x i32> <i32 1, i32 1>, <i32 1, i32 1>
  %res4 = add <2 x i32> %x, zeroinitializer
  ret void
}

define void @OtherConstants(i32 %x, i8* %Addr) {
entry:
  %res1 = add i32 %x, undef
  %poison = sub nuw i32 0, 1
  %res2 = icmp eq i8* blockaddress(@OtherConstants, %Next), null
  br label %Next

Next:                                             ; preds = %entry
  ret void
}

define void @OtherConstants2() {
entry:
  %0 = trunc i32 1 to i8
  %1 = zext i8 1 to i32
  %2 = sext i8 1 to i32
  %3 = fptrunc double 1.000000e+00 to float
  %4 = fpext float 1.000000e+00 to double
  %5 = fptosi float 1.000000e+00 to i32
  %6 = uitofp i32 1 to float
  %7 = sitofp i32 -1 to float
  %8 = ptrtoint i32* @X to i32
  %9 = inttoptr i8 1 to i8*
  %10 = bitcast i32 1 to <2 x i16>
  %11 = getelementptr i32, i32* @X, i32 0
  %12 = getelementptr inbounds i32, i32* @X, i32 0
  %13 = select i1 true, i32 1, i32 0
  %14 = icmp eq i32 1, 0
  %15 = fcmp oeq float 1.000000e+00, 0.000000e+00
  %16 = extractelement <2 x i32> <i32 1, i32 1>, i32 1
  %17 = insertelement <2 x i32> <i32 1, i32 1>, i32 0, i32 1
  %18 = shufflevector <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %19 = extractvalue { i32, float } { i32 1, float 2.000000e+00 }, 0
  %20 = insertvalue { i32, float } { i32 1, float 2.000000e+00 }, i32 0, 0
  ret void
}
