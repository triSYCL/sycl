; ModuleID = './llvm/bindings/python/llvm/tests/test.bc'
source_filename = "./llvm/bindings/python/llvm/tests/test.bc"
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%foobar = type { i32, [0 x i8] }
%zero2 = type {}
%zero2r = type { %zero2 }
%foobar2 = type { i32, %zero2r }

define void @f(%foobar %arg) {
  %arg1 = extractvalue %foobar %arg, 0
  %arg2 = extractvalue %foobar %arg, 1
  %1 = call i32 @f2([0 x i8] %arg2, i32 5, i32 42)
  ret void
}

define i32 @f2([0 x i8] %x, i32 %y, i32 %z) {
  ret i32 %y
}

define void @f3([0 x i8] %x, i32 %y) {
  %1 = call i32 @f2([0 x i8] %x, i32 5, i32 %y)
  ret void
}

define void @f4([0 x i8] %z) {
  %1 = insertvalue %foobar undef, [0 x i8] %z, 1
  ret void
}

define void @f5(%foobar %x) {
allocas:
  %y = extractvalue %foobar %x, 1
  br label %b1

b1:                                               ; preds = %allocas
  %insert120 = insertvalue %foobar undef, [0 x i8] %y, 1
  ret void
}

define void @f6([0 x i8] %x, [0 x i8] %y) {
b1:
  br i1 undef, label %end, label %b2

b2:                                               ; preds = %b1
  br label %end

end:                                              ; preds = %b2, %b1
  %z = phi [0 x i8] [ %y, %b1 ], [ %x, %b2 ]
  call void @f4([0 x i8] %z)
  ret void
}

define i32 @g1(%zero2 %x, i32 %y, i32 %z) {
  ret i32 %y
}

define void @g2(%zero2 %x, i32 %y) {
  %1 = call i32 @g1(%zero2 %x, i32 5, i32 %y)
  ret void
}

define i32 @h1(%zero2r %x, i32 %y, i32 %z) {
  ret i32 %y
}

define void @h2(%zero2r %x, i32 %y) {
  %1 = call i32 @h1(%zero2r %x, i32 5, i32 %y)
  ret void
}

define void @h3(%foobar2 %arg) {
  %arg1 = extractvalue %foobar2 %arg, 0
  %arg2 = extractvalue %foobar2 %arg, 1
  %arg21 = extractvalue %zero2r %arg2, 0
  call void @g2(%zero2 %arg21, i32 5)
  ret void
}
