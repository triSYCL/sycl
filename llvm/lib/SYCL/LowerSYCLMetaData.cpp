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
#include <iterator>
#include <regex>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/SYCL/LowerSYCLMetaData.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

namespace {

static StringRef kindOf(const char *Str) {
  return StringRef(Str, strlen(Str) + 1);
}

struct LSMDState {
  LSMDState(Module &M) : M(M), Ctx(M.getContext()) {}
  Module &M;
  LLVMContext &Ctx;
  bool HasChanged = false;

  llvm::SmallDenseMap<Function *, std::unique_ptr<DominatorTree>, 16> DTCache;
  llvm::SmallDenseMap<Function *, std::unique_ptr<LoopInfo>, 16> LICache;
  // llvm::SmallDenseSet<Function*, 8>

  Loop *findLoopAround(Instruction *I) {
    Function *F = I->getFunction();
    std::unique_ptr<LoopInfo> &LI = LICache[F];
    if (!LI) {
      std::unique_ptr<DominatorTree> &DT = DTCache[F];
      if (!DT)
        DT = std::make_unique<DominatorTree>(*F);
      LI = std::make_unique<LoopInfo>(*DT);
    }
    return LI->getLoopFor(I->getParent());
  }

  void findLoopAroundFunction(Function *Start,
                              function_ref<void(Loop *)> Functor) {
    llvm::SmallSetVector<Function *, 8> Stack;
    Stack.insert(Start);
    while (!Stack.empty()) {
      Function *F = Stack.pop_back_val();
      for (auto &U : F->uses()) {
        User *Usr = U.getUser();
        auto OnLoop = [&](Loop *L) {
          if (L)
            return Functor(L);
          if (auto *I = dyn_cast<Instruction>(Usr))
            Stack.insert(I->getFunction());
        };
        if (auto *CB = dyn_cast<CallBase>(Usr))
          if (U.getOperandNo() == CB->arg_size())
            OnLoop(findLoopAround(CB));

        // TODO: Try follow uses to deal with function pointers.
      }
    }
  }

  /// @brief Add HLS-compatible pipeline annotation to surrounding loop
  ///
  /// @param CS Payload of the original annotation
  void lowerPipeline(llvm::ConstantStruct *CS) {
    auto *F =
        dyn_cast<Function>(getUnderlyingObject(CS->getAggregateElement(0u)));

    if (!F)
      return;

    // 4th element of the annotation is the payload (first three are pointer
    // to annotated, source file, line number)
    auto *CSArgs = cast<Constant>(
        cast<GlobalVariable>(getUnderlyingObject(CS->getAggregateElement(4)))
            ->getOperand(0));
    auto *IIInitializer =
        cast<ConstantInt>(getUnderlyingObject(CSArgs->getAggregateElement(0u)));
    auto *RewindInitializer =
        cast<ConstantInt>(getUnderlyingObject(CSArgs->getAggregateElement(1u)));
    auto *PipelineType =
        cast<ConstantInt>(getUnderlyingObject(CSArgs->getAggregateElement(2u)));

    findLoopAroundFunction(F, [=](Loop *L) {
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
        // HLS pipeline take 3 values : the required II, a bool
        // indicating if the loop should rewind, and the pipeline type.
        ResultMD.push_back(MDNode::get(
            Ctx,
            {
                MDString::get(Ctx, "llvm.loop.pipeline.enable"),
                ConstantAsMetadata::get(ConstantInt::get(
                    Type::getInt32Ty(Ctx), IIInitializer->getSExtValue())),
                ConstantAsMetadata::get(ConstantInt::get(
                    Type::getInt1Ty(Ctx), RewindInitializer->getSExtValue())),
                ConstantAsMetadata::get(ConstantInt::get(
                    Type::getIntNTy(Ctx, 2), PipelineType->getSExtValue())),
            }));
        MDNode *MDN = MDNode::getDistinct(Ctx, ResultMD);
        BB->getTerminator()->setMetadata(LLVMContext::MD_loop, MDN);
        BB->getTerminator()
            ->getMetadata(LLVMContext::MD_loop)
            ->replaceOperandWith(0, MDN);
      }
    });
  }

  void lowerPipelineKernel(llvm::Function *F,
                           llvm::ConstantStruct *Parameters) {
    auto *IIInitializer = cast<ConstantInt>(
        getUnderlyingObject(Parameters->getAggregateElement(0u)));
    auto *PipelineType = cast<ConstantInt>(
        getUnderlyingObject(Parameters->getAggregateElement(1u)));
    std::string S = formatv("{0}.{1}", IIInitializer->getSExtValue(),
                            PipelineType->getSExtValue());
    F->addFnAttr("fpga.static.pipeline", S);
  }

  void lowerKernelParam(llvm::Function *F,
                           llvm::ConstantStruct *Parameters) {
    StringRef ExtrtaArgs =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(Parameters->getAggregateElement(0u)))
                ->getOperand(0))
            ->getRawDataValues();
    F->addFnAttr("fpga.vpp.extraargs", ExtrtaArgs);
  }

  void lowerArrayPartition(llvm::Value *V) {
    auto *F = dyn_cast<Function>(V);
    if (!F)
      return;

    HasChanged = true;
    Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    OperandBundleDef OpBundle("xlx_array_partition",
                              ArrayRef<Value *>{F->getArg(0), F->getArg(1),
                                                F->getArg(2), F->getArg(3)});

    Instruction *I = CallInst::Create(SideEffect, {}, {OpBundle});
    I->insertBefore(F->getEntryBlock().getTerminator());
  }

  void dispatchKernelPropertyToHandler(llvm::ConstantStruct *CS) {
    auto *F =
        dyn_cast<Function>(getUnderlyingObject(CS->getAggregateElement(0u)));

    if (!F)
      return;
    // 4th element of the annotation is the payload (first three are pointer
    // to annotated, source file, line number)
    auto *CSArgs = cast<Constant>(
        cast<GlobalVariable>(getUnderlyingObject(CS->getAggregateElement(4)))
            ->getOperand(0));
    // Property is always constituted by a string for the property type,
    // and a (possibly empty) struct for the payload
    StringRef PropertyType =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(CSArgs->getAggregateElement(0u)))
                ->getOperand(0))
            ->getRawDataValues();
    auto *PropertyPayload = cast<ConstantStruct>(
        getUnderlyingObject(CSArgs->getAggregateElement(1u)));
    bool isWrapper = false;
    if (PropertyType == "kernel_pipeline") {
      lowerPipelineKernel(F, PropertyPayload);
      isWrapper = true;
    } else if (PropertyType == "kernel_param") {
      lowerKernelParam(F, PropertyPayload);
      isWrapper = true;
    }

    if (isWrapper) {
      F->addFnAttr("fpga.propertywrapper", "true");
    }
  }

  /// @brief Check if a global annotation (from llvm.global.annotations)
  /// corresponds to a marker that has to be converted to an HLS-compatible
  /// annotation
  void processGlobalAnnotation(llvm::Value *Annot) {
    auto *CS = cast<ConstantStruct>(Annot);
    StringRef AnnotKind =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(CS->getAggregateElement(1)))
                ->getOperand(0))
            ->getRawDataValues();
    if (AnnotKind == kindOf("xilinx_pipeline")) {
      lowerPipeline(CS);
    } else if (AnnotKind == kindOf("xilinx_partition_array")) {
      lowerArrayPartition(getUnderlyingObject(CS->getAggregateElement(0u)));
    } else if (AnnotKind == kindOf("xilinx_kernel_property")) {
      dispatchKernelPropertyToHandler(CS);
    }

    return;
  }

  bool run() {
    GlobalVariable *Annots = M.getGlobalVariable("llvm.global.annotations");
    if (!Annots)
      return false;
    auto *Array = cast<ConstantArray>(Annots->getOperand(0));
    for (auto *E : Array->operand_values())
      processGlobalAnnotation(E);

    return HasChanged;
  }
};

struct LowerSYCLMetaData : public ModulePass {

  static char ID; // Pass identification, replacement for typeid
  LowerSYCLMetaData() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    bool GlobalChanges = LSMDState(M).run();
    return GlobalChanges;
  }
  virtual StringRef getPassName() const override { return "LowerSYCLMetaData"; }
};
} // namespace

namespace llvm {
void initializeLowerSYCLMetaDataPass(PassRegistry &Registry);
} // namespace llvm

INITIALIZE_PASS(LowerSYCLMetaData, "lower-sycl-metadata",
                "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createLowerSYCLMetaDataPass() {
  return new LowerSYCLMetaData();
}

char LowerSYCLMetaData::ID = 0;
