//===--- XilinxHLS.cpp - Implement XilinxHLS target feature support
//-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XilinxHLS TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "XilinxHLS.h"
#include "Targets.h"

using namespace clang;
using namespace clang::targets;

void XilinxHLSTargetInfo::getTargetDefines(const LangOptions &Opts,
                                           MacroBuilder &Builder) const {
  DefineStd(Builder, "XilinxHLS", Opts);
}

XilinxHLS32TargetInfo::XilinxHLS32TargetInfo(const llvm::Triple &Triple,
                                             const TargetOptions &Opts)
    : XilinxHLSTargetInfo(Triple, Opts) {
  PointerWidth = PointerAlign = 32;
  SizeType = TargetInfo::UnsignedInt;
  PtrDiffType = IntPtrType = TargetInfo::SignedInt;
  resetDataLayout(
      "e-m:e-p:32:32-"
      "i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-"
      "n8:16:32:64-S128-"
      "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
      "1024");
}

void XilinxHLS32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                             MacroBuilder &Builder) const {
  XilinxHLSTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "XilinxHLS32", Opts);
}

XilinxHLS64TargetInfo::XilinxHLS64TargetInfo(const llvm::Triple &Triple,
                                             const TargetOptions &Opts)
    : XilinxHLSTargetInfo(Triple, Opts) {
  PointerWidth = PointerAlign = 64;
  SizeType = TargetInfo::UnsignedLong;
  PtrDiffType = IntPtrType = TargetInfo::SignedLong;
  resetDataLayout(
      "e-m:e-"
      "i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-"
      "n8:16:32:64-S128-"
      "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
      "1024");
}

void XilinxHLS64TargetInfo::getTargetDefines(const LangOptions &Opts,
                                             MacroBuilder &Builder) const {
  XilinxHLSTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "XilinxHLS64", Opts);
  DefineStd(Builder, "x86_64", Opts);
}
