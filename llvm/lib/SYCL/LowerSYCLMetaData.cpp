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
#include <iostream>
#include <regex>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/SYCL/LowerSYCLMetaData.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

namespace {

static StringRef kindOf(const char* Str) {
  return StringRef(Str, strlen(Str) + 1);
}

class LocalAnnotationVisitor : public InstVisitor<LocalAnnotationVisitor> {
  Module &M;
  LLVMContext &C;
  bool HasChanged;

  void processMemBankAssign(CallInst &I) {
    HasChanged = true;
    Constant *Bank =
        (cast<GlobalVariable>(getUnderlyingObject(I.getOperand(4)))
             ->getInitializer());
    auto BankMem = isa<ConstantAggregateZero>(Bank)
                       ? 0
                       : cast<ConstantInt>(Bank->getOperand(0))->getZExtValue();
    auto BundleIdentifier = "ddrmem" + std::to_string(BankMem);
    auto *BundleIDConstant = ConstantDataArray::getString(C, BundleIdentifier, false);
    auto *MinusOne = ConstantInt::getSigned(IntegerType::get(C, 64), -1);
    auto *AnnotatedInstr = dyn_cast_or_null<AllocaInst>(getUnderlyingObject(I.getOperand(0)));
    Argument* Annotated = nullptr;
    for (Argument *Arg = I.getCaller()->arg_begin();
         Arg != I.getCaller()->arg_end(); ++Arg) {
      for (User *U : Arg->users()) {
        if (auto *Store = dyn_cast<StoreInst>(U))
          if (getUnderlyingObject(Store->getPointerOperand()) == AnnotatedInstr) {
            Annotated = Arg;
          }
      }
    }

    auto *CAZ = ConstantAggregateZero::get(ArrayType::get(IntegerType::get(C, 8), 0));
    Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    SideEffect->addFnAttr(Attribute::NoUnwind);
    SideEffect->addFnAttr(Attribute::InaccessibleMemOnly);
    //TODO find a clever default value, allow user customisation via properties
    SideEffect->addFnAttr("xlx.port.bitwidth", "4096");

    OperandBundleDef OpBundle(
        "xlx_m_axi", ArrayRef<Value *>{Annotated, BundleIDConstant, MinusOne,
                                       CAZ, CAZ, MinusOne, MinusOne, MinusOne,
                                       MinusOne, MinusOne, MinusOne});
    Instruction *Instr = CallInst::Create(SideEffect, {}, {OpBundle});
    Instr->insertBefore(I.getCaller()->getEntryBlock().getTerminator());
  }

  void processLocalAnnotation(IntrinsicInst &I) {
    StringRef Annotation =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(getUnderlyingObject(I.getOperand(1)))
                ->getInitializer())
            ->getRawDataValues();
    if (Annotation == kindOf("xilinx_ddr_bank")) {
      processMemBankAssign(I);
    }
  }

public:
  LocalAnnotationVisitor(Module &M) : M(M), C(M.getContext()), HasChanged(false) {}
  void visitIntrinsicInst(IntrinsicInst &I) {
    if (I.getIntrinsicID() != Intrinsic::var_annotation)
      return;
    processLocalAnnotation(I);
  }

  bool hasChanged() {
    return HasChanged;
  }
};

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



  void processGlobalAnnotation(llvm::Value *Annot) {
    auto* CS = cast<ConstantStruct>(Annot);
    StringRef AnnotKind =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(CS->getAggregateElement(1)))
                ->getOperand(0))
            ->getRawDataValues();
    if (AnnotKind == kindOf("xilinx_pipeline"))
      lowerPipeline(getUnderlyingObject(CS->getAggregateElement(0u)));
    else if (AnnotKind == kindOf("xilinx_partition_array"))
      lowerArrayPartition(getUnderlyingObject(CS->getAggregateElement(0u)));
    return;
  }

  bool run() {
    GlobalVariable *Annots = M.getGlobalVariable("llvm.global.annotations");
    if (!Annots)
      return false;
    
    auto * Array = cast<ConstantArray>(Annots->getOperand(0));
    for (auto* E : Array->operand_values())
      processGlobalAnnotation(E);

    return HasChanged;
  }
};

struct LowerSYCLMetaData : public ModulePass {

  static char ID; // Pass identification, replacement for typeid
  LowerSYCLMetaData() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    bool GlobalChanges = LSMDState(M).run();
    LocalAnnotationVisitor LV(M);
    LV.visit(M);
    bool LocalChanges = LV.hasChanged();
    return GlobalChanges or LocalChanges;
  }
  virtual StringRef getPassName() const override {
    return "LowerSYCLMetaData";
  }
};
} // namespace

namespace llvm {
void initializeLowerSYCLMetaDataPass(PassRegistry &Registry);
} // namespace llvm

INITIALIZE_PASS(LowerSYCLMetaData, "lower-sycl-metadata",
  "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createLowerSYCLMetaDataPass() {return new LowerSYCLMetaData();}

char LowerSYCLMetaData::ID = 0;
