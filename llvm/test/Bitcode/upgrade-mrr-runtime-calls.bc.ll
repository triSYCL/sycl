; ModuleID = './llvm/test/Bitcode/upgrade-mrr-runtime-calls.bc'
source_filename = "<stdin>"
target triple = "arm64-apple-ios7.0"

define void @testRuntimeCalls(i8* %a, i8** %b, i8** %c, i32* %d, i32** %e) personality i32 (...)* @__gxx_personality_v0 {
entry:
  %v0 = tail call i8* @objc_autorelease(i8* %a) #0
  tail call void @objc_autoreleasePoolPop(i8* %a) #0
  %v1 = tail call i8* @objc_autoreleasePoolPush() #0
  %v2 = tail call i8* @objc_autoreleaseReturnValue(i8* %a) #0
  tail call void @objc_copyWeak(i8** %b, i8** %c) #0
  tail call void @objc_destroyWeak(i8** %b) #0
  %v3 = tail call i32* @objc_initWeak(i32** %e, i32* %d) #0
  %v4 = tail call i8* @objc_loadWeak(i8** %b) #0
  %v5 = tail call i8* @objc_loadWeakRetained(i8** %b) #0
  tail call void @objc_moveWeak(i8** %b, i8** %c) #0
  tail call void @objc_release(i8* %a) #0
  %v6 = tail call i8* @objc_retain(i8* %a) #0
  %v7 = tail call i8* @objc_retainAutorelease(i8* %a) #0
  %v8 = tail call i8* @objc_retainAutoreleaseReturnValue(i8* %a) #0
  %v9 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %a) #0
  %v10 = tail call i8* @objc_retainBlock(i8* %a) #0
  tail call void @objc_storeStrong(i8** %b, i8* %a) #0
  %v11 = tail call i8* @objc_storeWeak(i8** %b, i8* %a) #0
  tail call void (...) @llvm.objc.clang.arc.use(i8* %a)
  %v12 = tail call i8* @objc_unsafeClaimAutoreleasedReturnValue(i8* %a) #0
  %v13 = tail call i8* @objc_retainedObject(i8* %a) #0
  %v14 = tail call i8* @objc_unretainedObject(i8* %a) #0
  %v15 = tail call i8* @objc_unretainedPointer(i8* %a) #0
  %v16 = tail call i8* @objc_retain.autorelease(i8* %a) #0
  %v17 = tail call i32 @objc_sync.enter(i8* %a) #0
  %v18 = tail call i32 @objc_sync.exit(i8* %a) #0
  tail call void @objc_arc_annotation_topdown_bbstart(i8** %b, i8** %c) #0
  tail call void @objc_arc_annotation_topdown_bbend(i8** %b, i8** %c) #0
  tail call void @objc_arc_annotation_bottomup_bbstart(i8** %b, i8** %c) #0
  tail call void @objc_arc_annotation_bottomup_bbend(i8** %b, i8** %c) #0
  invoke void @objc_autoreleasePoolPop(i8* %a)
          to label %normalBlock unwind label %unwindBlock

normalBlock:                                      ; preds = %entry
  ret void

unwindBlock:                                      ; preds = %entry
  %ll = landingpad { i8*, i32 }
          cleanup
  ret void
}

declare i8* @objc_autorelease(i8*)

declare void @objc_autoreleasePoolPop(i8*)

declare i8* @objc_autoreleasePoolPush()

declare i8* @objc_autoreleaseReturnValue(i8*)

declare void @objc_copyWeak(i8**, i8**)

declare void @objc_destroyWeak(i8**)

declare i32* @objc_initWeak(i32**, i32*)

declare i8* @objc_loadWeak(i8**)

declare i8* @objc_loadWeakRetained(i8**)

declare void @objc_moveWeak(i8**, i8**)

declare void @objc_release(i8*)

declare i8* @objc_retain(i8*)

declare i8* @objc_retainAutorelease(i8*)

declare i8* @objc_retainAutoreleaseReturnValue(i8*)

declare i8* @objc_retainAutoreleasedReturnValue(i8*)

declare i8* @objc_retainBlock(i8*)

declare void @objc_storeStrong(i8**, i8*)

declare i8* @objc_storeWeak(i8**, i8*)

declare i8* @objc_unsafeClaimAutoreleasedReturnValue(i8*)

declare i8* @objc_retainedObject(i8*)

declare i8* @objc_unretainedObject(i8*)

declare i8* @objc_unretainedPointer(i8*)

declare i8* @objc_retain.autorelease(i8*)

declare i32 @objc_sync.enter(i8*)

declare i32 @objc_sync.exit(i8*)

declare void @objc_arc_annotation_topdown_bbstart(i8**, i8**)

declare void @objc_arc_annotation_topdown_bbend(i8**, i8**)

declare void @objc_arc_annotation_bottomup_bbstart(i8**, i8**)

declare void @objc_arc_annotation_bottomup_bbend(i8**, i8**)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare void @llvm.objc.clang.arc.use(...) #0

attributes #0 = { nounwind }
