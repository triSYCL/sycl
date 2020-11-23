//===- PrepareSYCLOpt.cpp                                    ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Prepare device code for Optimizations.
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/SYCL/PrepareSYCLOpt.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

namespace {

struct PrepareSYCLOpt : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  PrepareSYCLOpt() : ModulePass(ID) {}

  void turnNonKernelsIntoPrivate(Module &M) {
    for (GlobalObject &G : M.global_objects()) {
      if (auto *F = dyn_cast<Function>(&G))
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL)
          continue;
      if (G.isDeclaration())
        continue;
      G.setComdat(nullptr);
      G.setLinkage(llvm::GlobalValue::PrivateLinkage);
    }
  }

  void setCallingConventions(Module& M) {
    for (Function &F : M.functions()) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
        assert(F.use_empty());
        continue;
      }
      if (F.isIntrinsic())
        continue;
      F.setCallingConv(CallingConv::SPIR_FUNC);
      for (Value* V : F.users()) {
        if (auto* Call = dyn_cast<CallBase>(V))
          Call->setCallingConv(CallingConv::SPIR_FUNC);
      }
    }
  }

  /// At this point in the pipeline Annotations intrinsic have all been
  /// converted into what they need to be. But they can still be present and
  /// have pointer on pointer as arguments which v++ can't deal with.
  void removeAnnotations(Module &M) {
    SmallVector<Instruction *, 16> ToRemove;
    for (Function &F : M.functions())
      if (F.getIntrinsicID() == Intrinsic::annotation ||
          F.getIntrinsicID() == Intrinsic::ptr_annotation ||
          F.getIntrinsicID() == Intrinsic::var_annotation)
        for (User *U : F.users())
          if (auto *I = dyn_cast<Instruction>(U))
            ToRemove.push_back(I);
    for (Instruction *I : ToRemove)
      I->eraseFromParent();
    GlobalVariable *Annot = M.getGlobalVariable("llvm.global.annotations");
    if (Annot)
      Annot->eraseFromParent();
  }

  /// This will change array partition such that after the O3 pipeline it
  /// matched very closely what v++ generates.
  /// This will change the type of the alloca referenced by the array partition
  /// into an array. and change the argument received by xlx_array_partition
  /// into a pointer on an array.
  void lowerArrayPartition(Module &M) {
    Function* Func = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    for (Use& U : Func->uses()) {
      auto* Usr = dyn_cast<CallBase>(U.getUser());
      if (!Usr)
        continue;
      if (!Usr->getOperandBundle("xlx_array_partition"))
        continue;
      Use& Ptr = U.getUser()->getOperandUse(0);
      Value* Obj = getUnderlyingObject(Ptr);
      if (!isa<AllocaInst>(Obj))
        return;
      auto* Alloca = cast<AllocaInst>(Obj);
      auto *Replacement =
          new AllocaInst(Ptr->getType()->getPointerElementType(), 0,
                         ConstantInt::get(Type::getInt32Ty(M.getContext()), 1),
                         Align(128), "");
      Replacement->insertAfter(Alloca);
      Instruction* Cast = BitCastInst::Create(
          Instruction::BitCast, Replacement, Alloca->getType());
      Cast->insertAfter(Replacement);
      Alloca->replaceAllUsesWith(Cast);
      Value* Zero = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
      Instruction* GEP = GetElementPtrInst::Create(nullptr, Replacement, {Zero});
      GEP->insertAfter(Cast);
      Ptr.set(GEP);
    }
  }

  bool runOnModule(Module &M) override {
    turnNonKernelsIntoPrivate(M);
    setCallingConventions(M);
    lowerArrayPartition(M);
    removeAnnotations(M);
    return true;
  }
};
}

namespace llvm {
void initializePrepareSYCLOptPass(PassRegistry &Registry);
}

INITIALIZE_PASS(PrepareSYCLOpt, "preparesycl",
  "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createPrepareSYCLOptPass() {return new PrepareSYCLOpt();}

char PrepareSYCLOpt::ID = 0;
