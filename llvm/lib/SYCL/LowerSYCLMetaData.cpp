//===- LowerSYCLMetaData.cpp ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Lower optimization metadata. This Pass must be run before inling.
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Instructions.h"
#include "llvm/SYCL/LowerSYCLMetaData.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace llvm;

namespace {

struct LSMDState {
  LSMDState(Module& M) : M(M), Ctx(M.getContext()) {}
  Module& M;
  LLVMContext& Ctx;
  bool HasChanged = false;

  llvm::SmallDenseMap<Function*, std::unique_ptr<DominatorTree>, 16> DTCache;
  llvm::SmallDenseMap<Function*, std::unique_ptr<LoopInfo>, 16> LICache;
  // llvm::SmallDenseSet<Function*, 8>

  Loop* findLoopAround(Instruction* I) {
    Function* F = I->getFunction();
    std::unique_ptr<LoopInfo>& LI = LICache[F];
    if (!LI) {
      std::unique_ptr<DominatorTree> &DT = DTCache[F];
      if (!DT)
        DT = std::make_unique<DominatorTree>(*F);
      LI = std::make_unique<LoopInfo>(*DT);
    }
    return LI->getLoopFor(I->getParent());
  }

  void findLoopAroundFunction(Function *Start, function_ref<void(Loop *)> Functor) {
    llvm::SmallSetVector<Function *, 8> Stack;
    Stack.insert(Start);
    while (!Stack.empty()) {
      Function *F = Stack.pop_back_val();
      for (auto &U : F->uses()) {
        User *Usr = U.getUser();
        auto OnLoop = [&](Loop* L) {
          if (L)
            return Functor(L);
          if (auto* I = dyn_cast<Instruction>(Usr))
            Stack.insert(I->getFunction());
        };
        if (auto *CB = dyn_cast<CallBase>(Usr))
          if (U.getOperandNo() == CB->arg_size())
            OnLoop(findLoopAround(CB));
        
        //TODO: Try follow uses to deal with function pointers.
      }
    }
  }

  void lowerPipeline(llvm::Value* V) {
    auto* F = dyn_cast<Function>(V);
    if (!F)
      return;
    findLoopAroundFunction(F, [this](Loop *L) {
      SmallVector<BasicBlock *, 4> LoopLatches;
      L->getLoopLatches(LoopLatches);
      for (BasicBlock *BB : LoopLatches) {
        SmallVector<Metadata *, 4> ResultMD;
        if (BB->getTerminator()->hasMetadata(LLVMContext::MD_loop))
          ResultMD.append(
              BB->getTerminator()
                  ->getMetadata(LLVMContext::MD_loop)
                  ->op_begin(),
              BB->getTerminator()->getMetadata(LLVMContext::MD_loop)->op_end());
        else
          ResultMD.push_back(MDNode::getTemporary(Ctx, None).get());
        ResultMD.push_back(MDNode::get(
            Ctx, {
                     MDString::get(Ctx, "llvm.loop.pipeline.enable"),
                     ConstantAsMetadata::get(
                         ConstantInt::get(Type::getInt32Ty(Ctx), -1)),
                     ConstantAsMetadata::get(
                         ConstantInt::getFalse(Type::getInt1Ty(Ctx))),
                     ConstantAsMetadata::get(
                         ConstantInt::get(Type::getInt32Ty(Ctx), 0)),
                 }));
        MDNode *MDN = MDNode::getDistinct(Ctx, ResultMD);
        BB->getTerminator()->setMetadata(LLVMContext::MD_loop, MDN);
        BB->getTerminator()
            ->getMetadata(LLVMContext::MD_loop)
            ->replaceOperandWith(0, MDN);
      }
    });
  }

  void lowerArrayPartition(llvm::Value* V) {
    auto *F = dyn_cast<Function>(V);
    if (!F)
      return;

    HasChanged = true;
    Function* SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    OperandBundleDef OpBundle("xlx_array_partition",
                              ArrayRef<Value *>{F->getArg(0), F->getArg(1),
                                                F->getArg(2), F->getArg(3)});

    Instruction* I = CallInst::Create(SideEffect, {}, {OpBundle});
    I->insertBefore(F->getEntryBlock().getTerminator());
  }

  static StringRef KindOf(const char* Str) {
    return StringRef(Str, strlen(Str) + 1);
  }

  void processAnnotation(llvm::Value *Annot) {
    auto* CS = cast<ConstantStruct>(Annot);
    StringRef AnnotKind =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(CS->getAggregateElement(1)))
                ->getOperand(0))
            ->getRawDataValues();
    if (AnnotKind == KindOf("xilinx_pipeline"))
      lowerPipeline(getUnderlyingObject(CS->getAggregateElement(0u)));
    else if (AnnotKind == KindOf("xilinx_partition_array"))
      lowerArrayPartition(getUnderlyingObject(CS->getAggregateElement(0u)));
    return;
  }

  bool run() {
    GlobalVariable *Annots = M.getGlobalVariable("llvm.global.annotations");
    if (!Annots)
      return false;
    
    auto * Array = cast<ConstantArray>(Annots->getOperand(0));
    for (auto* E : Array->operand_values())
      processAnnotation(E);

    return HasChanged;
  }
};

struct LowerSYCLMetaData : public ModulePass {

  static char ID; // Pass identification, replacement for typeid
  LowerSYCLMetaData() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    return LSMDState(M).run();
  }
  virtual StringRef getPassName() const override {
    return "LowerSYCLMetaData";
  }
};
}

namespace llvm {
void initializeLowerSYCLMetaDataPass(PassRegistry &Registry);
}

INITIALIZE_PASS(LowerSYCLMetaData, "lower-sycl-metadata",
  "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createLowerSYCLMetaDataPass() {return new LowerSYCLMetaData();}

char LowerSYCLMetaData::ID = 0;
