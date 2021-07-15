//===- ChessMassage.h - SYCL chess-clang IR Downgrader pass  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass for modifying certain LLVM IR incompatabilities with the chess-clang
// backend we use for SYCL
//
// ===---------------------------------------------------------------------===//

#ifndef LLVM_SYCL_CHESS_MASSAGE_H
#define LLVM_SYCL_CHESS_MASSAGE_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createChessMassagePass();

}

#endif
