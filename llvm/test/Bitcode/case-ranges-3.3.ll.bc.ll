; ModuleID = './llvm/test/Bitcode/case-ranges-3.3.ll.bc'
source_filename = "./llvm/test/Bitcode/case-ranges-3.3.ll.bc"

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i32 %x) #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 %x, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  switch i32 %3, label %9 [
    i32 -3, label %4
    i32 -2, label %4
    i32 -1, label %4
    i32 0, label %4
    i32 1, label %4
    i32 2, label %4
    i32 4, label %5
    i32 5, label %6
    i32 6, label %7
    i32 7, label %8
  ]

4:                                                ; preds = %0, %0, %0, %0, %0, %0
  store i32 -1, i32* %1, align 4
  br label %11

5:                                                ; preds = %0
  store i32 2, i32* %1, align 4
  br label %11

6:                                                ; preds = %0
  store i32 1, i32* %1, align 4
  br label %11

7:                                                ; preds = %0
  store i32 4, i32* %1, align 4
  br label %11

8:                                                ; preds = %0
  store i32 3, i32* %1, align 4
  br label %11

9:                                                ; preds = %0
  br label %10

10:                                               ; preds = %9
  store i32 0, i32* %1, align 4
  br label %11

11:                                               ; preds = %10, %8, %7, %6, %5, %4
  %12 = load i32, i32* %1, align 4
  ret i32 %12
}

attributes #0 = { nounwind ssp uwtable }
