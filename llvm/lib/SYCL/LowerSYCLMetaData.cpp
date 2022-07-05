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
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/SYCL/LowerSYCLMetaData.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "SYCLUtils.h"

#define DEBUG_TYPE "lower-sycl-md"

using namespace llvm;

namespace {
cl::opt<bool> AfterO3("lower-delayed-sycl-metadata", cl::Hidden,
                      cl::init(false));

cl::opt<bool> ArrayPartitionHasModeArg("sycl-vxx-array-partition-mode-arg",
                                       cl::ReallyHidden);

static StringRef kindOf(const char *Str) {
  return StringRef(Str, strlen(Str) + 1);
}

struct LSMDState {
private:
  /// @brief Add annotation on Loop backedge (where HLS search for it)
  void annotateLoop(Loop *L, MDTuple *Annotation) {
    SmallVector<BasicBlock *, 4> LoopLatches;
    L->getLoopLatches(LoopLatches);
    for (BasicBlock *BB : LoopLatches) {
      SmallVector<Metadata *, 4> ResultMD;
      if (BB->getTerminator()->hasMetadata(LLVMContext::MD_loop))
        ResultMD.append(
            BB->getTerminator()->getMetadata(LLVMContext::MD_loop)->op_begin(),
            BB->getTerminator()->getMetadata(LLVMContext::MD_loop)->op_end());
      else
        ResultMD.push_back(MDNode::getTemporary(Ctx, None).get());

      ResultMD.push_back(Annotation);
      MDNode *MDN = MDNode::getDistinct(Ctx, ResultMD);
      BB->getTerminator()->setMetadata(LLVMContext::MD_loop, MDN);
      BB->getTerminator()
          ->getMetadata(LLVMContext::MD_loop)
          ->replaceOperandWith(0, MDN);
    }

    HasChanged = true;
  }

public:
  LSMDState(Module &M) : M(M), Ctx(M.getContext()) {}
  Module &M;
  LLVMContext &Ctx;
  bool HasChanged = false;

  llvm::SmallDenseMap<Function *, std::unique_ptr<DominatorTree>, 16> DTCache;
  llvm::SmallDenseMap<Function *, std::unique_ptr<LoopInfo>, 16> LICache;

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

  void applyOnEnclosingLoop(Function *Start,
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
  void lowerPipelineDecoration(llvm::ConstantStruct *CS) {
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

    applyOnEnclosingLoop(F, [=](Loop *L) {
      annotateLoop(
          L,
          MDNode::get(
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
    });
  }

  /// Lower xilinx_pipeline annotation attributes into HLS's representation for
  /// pipeline
  void lowerPipelineKernelDecoration(llvm::Function *F, llvm::Value *Payload) {
    std::string S;
    if (auto *Parameters = dyn_cast<ConstantStruct>(Payload)) {
      auto *IIInitializer = cast<ConstantInt>(
          getUnderlyingObject(Parameters->getAggregateElement(0u)));
      auto *PipelineType = cast<ConstantInt>(
          getUnderlyingObject(Parameters->getAggregateElement(1u)));
      S = formatv("{0}.{1}", IIInitializer->getSExtValue(),
                  PipelineType->getSExtValue());
    } else {
      S = "0.0";
    }
    F->addFnAttr("fpga.static.pipeline", S);
  }

  void lowerKernelParam(llvm::Function *F, llvm::Value *Payload) {
    auto *Parameters = cast<ConstantStruct>(Payload);
    StringRef ExtrtaArgs =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(Parameters->getAggregateElement(0u)))
                ->getOperand(0))
            ->getRawDataValues();
    F->addFnAttr("fpga.vpp.extraargs", ExtrtaArgs);
  }

  void lowerDataflowDecoration(llvm::ConstantStruct *CS) {
    auto *F =
        dyn_cast<Function>(getUnderlyingObject(CS->getAggregateElement(0u)));

    if (!F)
      return;

    applyOnEnclosingLoop(F, [=](Loop *L) {
      annotateLoop(
          L,
          MDNode::get(Ctx, {
                               MDString::get(Ctx, "llvm.loop.dataflow.enable"),
                           }));
    });
  }

  void lowerDataflowKernelDecoration(llvm::Function *F, llvm::Value *) {
    F->addFnAttr("fpga.dataflow.func", "0");
  }

  /// @brief Add HLS-compatible pipeline annotation to surrounding loop
  ///
  /// @param CS Payload of the original annotation
  void lowerUnrollDecoration(llvm::CallBase &CB,
                             llvm::ConstantStruct *Payload) {
    auto *F = CB.getCaller();

    // Metadata payload is unroll factor (first argument) and boolean indicating
    // whether the unrolling should be checked (in case of iteration not a
    // multiple of the unroll factor). Using u suffix to avoid ambiguity with
    // overloads of getAggregateElement.
    auto UnrollFactor =
        cast<ConstantInt>(getUnderlyingObject(Payload->getAggregateElement(0u)))
            ->getZExtValue();
    auto CheckUnroll =
        cast<ConstantInt>(getUnderlyingObject(Payload->getAggregateElement(1u)))
            ->getZExtValue();

    LoopInfo LI{DominatorTree{*F}};

    auto *Parent = CB.getParent();
    auto *EnclosingLoop = LI.getLoopFor(Parent);
    assert(EnclosingLoop != nullptr &&
           "Unroll annotation is not enclosed in loop");

    MDTuple *Annot;
    if (UnrollFactor == 0) {
      // Full unrolling
      Annot = MDNode::get(Ctx, {
                                   MDString::get(Ctx, "llvm.loop.unroll.full"),
                               });
    } else if (CheckUnroll) {
      // Checked partial unrolling
      Annot = MDNode::get(Ctx, {
                                   MDString::get(Ctx, "llvm.loop.unroll.count"),
                                   ConstantAsMetadata::get(ConstantInt::get(
                                       Type::getInt32Ty(Ctx), UnrollFactor)),
                               });
    } else {
      // Unchecked partial unrolling
      Annot = MDNode::get(
          Ctx, {
                   MDString::get(Ctx, "llvm.loop.unroll.withoutcheck"),
                   ConstantAsMetadata::get(
                       ConstantInt::get(Type::getInt32Ty(Ctx), UnrollFactor)),
               });
    }

    annotateLoop(EnclosingLoop, Annot);
  }

  template<typename ...Ts>
  void lowerAsSideEffect(
      llvm::Value *V, StringRef XclId, Ts... ts) {
    auto *F = dyn_cast<Function>(V);
    if (!F)
      return;

    HasChanged = true;
    Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    std::vector<Value *> Args;
    for (auto &A : F->args())
      Args.push_back(&A);
    std::ignore = std::initializer_list<int>{(Args.push_back(ts), 0)...};
    OperandBundleDef OpBundle(XclId.str(), Args);

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
    // Property is always constituted by a string for the property type (first
    // argument), and a payload (second argument)
    StringRef PropertyType =
        cast<ConstantDataArray>(
            cast<GlobalVariable>(
                getUnderlyingObject(CSArgs->getAggregateElement(0u)))
                ->getOperand(0))
            ->getRawDataValues();
    auto *PropertyPayload =
        getUnderlyingObject(CSArgs->getAggregateElement(1u));
    bool IsWrapper = false;
    if (!AfterO3) {
      if (PropertyType == "kernel_pipeline") {
        lowerPipelineKernelDecoration(F, PropertyPayload);
        IsWrapper = true;
      } else if (PropertyType == "kernel_param") {
        lowerKernelParam(F, PropertyPayload);
        IsWrapper = true;
      } else if (PropertyType == "kernel_dataflow") {
        lowerDataflowKernelDecoration(F, PropertyPayload);
        IsWrapper = true;
      }

      if (IsWrapper) {
        F->addFnAttr("fpga.propertywrapper", "true");
      }
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
    if (AnnotKind == kindOf("xilinx_kernel_property")) {
      dispatchKernelPropertyToHandler(CS);
    } else if (AnnotKind == kindOf("vitis_kernel")) {
      sycl::annotateKernelFunc(
          cast<Function>(getUnderlyingObject(CS->getAggregateElement(0u))));
    } else if (!AfterO3) { // Annotations that should be lowered before -O3
      if (AnnotKind == kindOf("xilinx_pipeline")) {
        lowerPipelineDecoration(CS);
      } else if (AnnotKind == kindOf("xilinx_partition_array")) {
        if (ArrayPartitionHasModeArg)
          lowerAsSideEffect(getUnderlyingObject(CS->getAggregateElement(0u)),
                            "xlx_array_partition", ConstantInt::getFalse(Ctx));
        else
          lowerAsSideEffect(getUnderlyingObject(CS->getAggregateElement(0u)),
                            "xlx_array_partition");
      } else if (AnnotKind == kindOf("xilinx_bind_storage")) {
        lowerAsSideEffect(getUnderlyingObject(CS->getAggregateElement(0u)),
                          "xlx_bind_storage");
      } else if (AnnotKind == kindOf("xilinx_dataflow")) {
        lowerDataflowDecoration(CS);
      }
    }
  }

  void dispatchLocalAnnotation(llvm::CallBase &CB,
                               llvm::ConstantDataArray *KindInit,
                               llvm::Constant *PayloadCst) {
    auto Kind = KindInit->getRawDataValues();
    bool processed = false;
    if (Kind == kindOf("xilinx_ddr_bank") || Kind == kindOf("xilinx_hbm_bank"))
      return;
    auto *Payload = cast<ConstantStruct>(PayloadCst);
    if (AfterO3) { // Annotation that should wait after optimisation to be
                   // lowered
      if (Kind == kindOf("xilinx_unroll")) {
        lowerUnrollDecoration(CB, Payload);
        processed = true;
      }
    }
    if (processed) {
      CB.eraseFromParent();
      HasChanged = true;
    }
  }

  static bool isFunc(Function *F, std::string pattern) {
    if (!F)
      return false;
    auto unmangled_name = llvm::demangle(F->getName().str());
    if (unmangled_name.find(pattern) != std::string::npos)
      return true;
    return false;
  }

  struct LocalAnnotationVisitor
      : public llvm::InstVisitor<LocalAnnotationVisitor> {
    LSMDState &Parent;
    LLVMContext &Ctx;
    IRBuilder<> Builder;

    struct PipeCreationT {
      llvm::Function *Func;
      llvm::CallBase *CB;
    };
    llvm::SmallVector<PipeCreationT> PipeCreations;

    /// Map the original version of a function to its most recent version.
    /// it should be used via getMostRecent
    SmallDenseMap<Function *, Function *> MostRecentFunc;

    /// Collection of functions to "delete" we dont do the deleting but allow
    /// global DCE to do it
    SmallVector<Function *> ToDelete;

    LocalAnnotationVisitor(LSMDState &P)
        : Parent(P), Ctx(Parent.Ctx), Builder(Ctx) {}

    void visitCallBase(CallBase &CB) {
      // Search for var_annotation having payload
      if (CB.getIntrinsicID() != Intrinsic::var_annotation &&
          CB.getCalledFunction()) {
        if (isFunc(CB.getCalledFunction(), "CreatePipeFromPipeStorage_")) {
          PipeCreations.emplace_back(PipeCreationT{CB.getFunction(), &CB});
        }
        return;
      }
      // Annotation without payload are not interesting
      if (CB.arg_size() < 5)
        return;

      auto *KindInit =
          cast<GlobalVariable>(getUnderlyingObject(CB.getOperand(1)))
              ->getInitializer();
      if (!isa<ConstantDataArray>(KindInit))
        return;
      auto *Payload = cast<Constant>(
          cast<GlobalVariable>(getUnderlyingObject(CB.getOperand(4)))
              ->getInitializer());
      Parent.dispatchLocalAnnotation(CB, cast<ConstantDataArray>(KindInit),
                                     Payload);
    }

    Function *&getMostRecent(Function *Func) {
      assert(Func);
      Function *&res = MostRecentFunc[Func];
      if (!res)
        res = Func;
      return res;
    }

    /// Return a string unique to the provided Type
    /// It is used to make Vitis HLS's function unique per type
    std::string getUniqueTypeStr(Type *T) {
      std::string str;
      raw_string_ostream os(str);
      T->print(os);
      if (T->isSingleValueType())
        return "b." + str;
      return "u." + str;
    }

    Function *getPipeReadFunc(Type *ValTy) {
      Type *PipeTy = PointerType::get(ValTy, 0);
      std::string FuncName =
          llvm::formatv("llvm.fpga.fifo.pop.{0}", getUniqueTypeStr(ValTy));
      FunctionType *FTy = FunctionType::get(ValTy, {PipeTy}, false);
      Function *Func = cast<Function>(
          Parent.M.getOrInsertFunction(FuncName, FTy).getCallee());
      return Func;
    }

    Function *getPipeWriteFunc(Type *ValTy) {
      Type *PipeTy = PointerType::get(ValTy, 0);
      std::string FuncName =
          llvm::formatv("llvm.fpga.fifo.push.{0}", getUniqueTypeStr(ValTy));
      FunctionType *FTy =
          FunctionType::get(Type::getVoidTy(Ctx), {ValTy, PipeTy}, false);
      Function *Func = cast<Function>(
          Parent.M.getOrInsertFunction(FuncName, FTy).getCallee());
      return Func;
    }

    bool isReadPipe(CallBase *CB) {
      return isFunc(CB->getCalledFunction(), "ReadPipeBlockingINTEL");
    }

    bool isWritePipe(CallBase *CB) {
      return isFunc(CB->getCalledFunction(), "WritePipeBlockingINTEL");
    }

    CallBase *replaceReadPipe(CallBase *Old, Value *Pipe) {
      assert(isReadPipe(Old));
      Type *ValTy = Pipe->getType()->getPointerElementType();
      auto *Val = Builder.CreateCall(getPipeReadFunc(ValTy), {Pipe});
      Builder.CreateStore(Val, Old->getArgOperand(1));
      Old->eraseFromParent();
      return Val;
    }

    CallBase *replaceWritePipe(CallBase *Old, Value *Pipe) {
      assert(isWritePipe(Old));
      Type *ValTy = Pipe->getType()->getPointerElementType();
      auto *Load = Builder.CreateLoad(ValTy, Old->getArgOperand(1));
      auto *NewCall = Builder.CreateCall(getPipeWriteFunc(ValTy), {Load, Pipe});
      Old->eraseFromParent();
      return NewCall;
    }

    Type *detectPipeType(CallBase *CB) {
      assert(!CB->user_empty());
      auto *Usage = *(CB->user_begin());
      CallBase *UsageCb = dyn_cast<CallBase>(Usage);
      assert(UsageCb != nullptr);
      llvm::Type *PipeType =
          UsageCb->getArgOperand(1)->getType()->getPointerElementType();
      return PipeType;
    }

    llvm::Function *
    duplicateFunctionWithExtraArgs(llvm::Function *Orig, llvm::Function *Func,
                                   ArrayRef<Type *> ExtraTypes,
                                   llvm::ValueToValueMapTy &ValMap) {
      assert(getMostRecent(Orig) == Func);
      llvm::FunctionType *FuncType = Func->getFunctionType();
      llvm::SmallVector<llvm::Type *> FuncArgs;
      for (auto *Type : FuncType->params())
        FuncArgs.push_back(Type);
      for (auto *Type : ExtraTypes)
        FuncArgs.push_back(PointerType::get(Type, 0));
      auto *NewFuncType =
          FunctionType::get(FuncType->getReturnType(), FuncArgs, false);

      std::string Fname = Func->getName().str();
      Func->setName(Fname + ".old");
      auto *NewFunc = Function::Create(NewFuncType, Func->getLinkage(), Fname,
                                       Func->getParent());
      sycl::giveNameToArguments(*NewFunc);

      for (size_t i = 0; i < Func->arg_size(); ++i)
        ValMap.insert({Func->getArg(i), NewFunc->getArg(i)});

      llvm::SmallVector<llvm::ReturnInst *> RetInst;
      llvm::CloneFunctionInto(NewFunc, Func, ValMap,
                              CloneFunctionChangeType::GlobalChanges, RetInst);
      MostRecentFunc[Orig] = NewFunc;
      return NewFunc;
    }

    void insertStreamInterfaceSideEffect(Argument *Pipe) {
      Function *SideEffect = Intrinsic::getDeclaration(
          Pipe->getParent()->getParent(), Intrinsic::sideeffect);
      OperandBundleDef OpBundle("stream_interface", ArrayRef<Value *>{Pipe});

      Instruction *I = CallInst::Create(SideEffect, {}, {OpBundle});
      I->insertBefore(&*Pipe->getParent()->getEntryBlock().begin());
    }

    void traverseCallGraph(llvm::Function *Func,
                           function_ref<void(CallBase *)> OnCall = nullptr,
                           function_ref<void(Function *)> OnFunc = nullptr) {
      llvm::DenseSet<Function *> UniqueEdge;
      SmallVector<Function *> ToBeProcessed;
      ToBeProcessed.push_back(Func);
      while (!ToBeProcessed.empty()) {
        Function *Curr = ToBeProcessed.pop_back_val();
        if (OnFunc && Curr != Func)
          OnFunc(Curr);
        if (sycl::isKernelFunc(Curr)) {
          continue;
        }
        UniqueEdge.clear();
        for (User *user : llvm::make_early_inc_range(Curr->users()))
          if (auto *CB = dyn_cast<CallBase>(user)) {
            if (UniqueEdge.insert(CB->getFunction()).second)
              ToBeProcessed.push_back(CB->getFunction());
            if (OnCall)
              OnCall(CB);
          }
      }
    }

    void handlePipeCreations(llvm::Function *Orig,
                             llvm::ArrayRef<CallBase *> Creations) {
      /// Most of the implementation for interprocedural rewrites is done but
      /// since it is rarely useful and not properly tested it is disabled
      assert(sycl::isKernelFunc(Orig) &&
             "interprocedural rewrites are not yet supported");
      Function *Func = getMostRecent(Orig);

      struct PipeInfo {
        bool isRead;
        int Depth;
        std::string ID;
      };

      /// Types of arguments representing pipes to we are adding to the function
      llvm::SmallVector<llvm::Type *> ExtraTypes;
      /// Information about the pipes we are adding used during interprocedural
      /// rewrites
      llvm::SmallVector<PipeInfo> PipeInfos;

      /// Add the correct pipe annotation iff Pipe is an argument of the
      /// top-level kernel function
      auto maybeAnnotatePipe = [&](int idx, Argument *Pipe) {
        if (sycl::isKernelFunc(Pipe->getParent())) {
          insertStreamInterfaceSideEffect(Pipe);
          PipeInfos[idx];
          if (PipeInfos[idx].isRead)
            sycl::annotateReadPipe(Pipe, PipeInfos[idx].ID,
                                   PipeInfos[idx].Depth);
          else
            sycl::annotateWritePipe(Pipe, PipeInfos[idx].ID,
                                    PipeInfos[idx].Depth);
        }
      };

      /// Collect the types of the pipes in the function
      for (auto *CB : Creations)
        ExtraTypes.push_back(detectPipeType(CB));

      /// Generate the new function with the extra augments
      /// But this function will still have the old IR
      llvm::ValueToValueMapTy ValMap;
      auto *NewFunc =
          duplicateFunctionWithExtraArgs(Orig, Func, ExtraTypes, ValMap);

      /// Update the body of the new function
      for (size_t i = 0; i < Creations.size(); ++i) {
        /// extra arguments are added to the end so we only look at arguments
        /// after Func->arg_size()
        Argument *Pipe = NewFunc->getArg(Func->arg_size() + i);

        auto OldCreate = Creations[i];
        /// the ValMap maps Values from the old function into the new one.
        /// We lookup into it to find the new Creation
        auto *newCB = cast<CallBase>(ValMap[OldCreate]);

        PipeInfo PInfo;
        /// The storage in Intel pipes is represented by a global variable
        /// provided to the call of CreatePipeFromPipeStorage_*
        GlobalVariable *PipeStorage =
            cast<GlobalVariable>(OldCreate->getArgOperand(0));

        /// The storage is unique to 1 pipe and the name of all global variable
        /// must be distinct so the name of the storage is unique to the pipe.
        /// and is used as identifier to by the pipe connection later.
        PInfo.ID = PipeStorage->getName().str();
        PInfo.Depth = cast<ConstantInt>(
                          cast<ConstantStruct>(PipeStorage->getInitializer())
                              ->getOperand(2))
                          ->getSExtValue();

        Builder.SetInsertPoint(newCB);
        bool hasRead = false;
        bool hasWrite = false;

        /// We remove the instructions while we iterate over them so we use
        /// llvm::make_early_inc_range
        for (auto *User : llvm::make_early_inc_range(newCB->users())) {
          /// Here we replace SPIRV intrinsics to read and write to pipes by
          /// ours (the one of Vitis HLS)
          auto *CB = cast<CallBase>(User);
          if (isReadPipe(CB)) {
            hasRead = true;
            replaceReadPipe(CB, Pipe);
          } else {
            hasWrite = true;
            replaceWritePipe(CB, Pipe);
          }

          assert(hasRead ^ hasWrite &&
                 "cannot read and write in the same pipe in the same kernel");
          PInfo.isRead = hasRead;
          PipeInfos.push_back(PInfo);

          /// Add annotation iff we are not doing and interprocedural rewrite
          maybeAnnotatePipe(i, Pipe);
        }
        newCB->eraseFromParent();
      }

      /// Rewrite all the functions and calls that can transitively call Func
      /// Collection of Nodes to modify
      SmallVector<Function *> FunctionToDuplicate;
      /// Collection of Edges to modify
      SmallVector<CallBase *> CallsToRewrite;

      /// Fill the collections
      traverseCallGraph(
          Func, [&](CallBase *CB) { CallsToRewrite.push_back(CB); },
          [&](Function *Caller) { FunctionToDuplicate.push_back(Caller); });

      /// Duplicate the Functions
      for (Function *Caller : FunctionToDuplicate) {
        Function *newCaller = getMostRecent(Caller);

        newCaller = duplicateFunctionWithExtraArgs(Caller, newCaller,
                                                   ExtraTypes, ValMap);
        for (unsigned i = 0; i < ExtraTypes.size(); i++) {
          Argument *Arg =
              newCaller->getArg(i + newCaller->arg_size() - ExtraTypes.size());
          /// Add annotation iff we are the top-level kernel
          maybeAnnotatePipe(i, Arg);
        }
        ToDelete.push_back(Caller);
      }

      /// Replace calls to the old function by calls to the new function
      for (CallBase *CB : CallsToRewrite) {
        /// Lookup the ValMap to get the call in the new function to an old
        /// function
        auto *CBInNewFunc = cast<CallBase>(ValMap[CB]);

        Function *newCaller = getMostRecent(CB->getFunction());
        /// Get the new function we need to call
        Function *newCallee = getMostRecent(CB->getCalledFunction());

        /// Build a param list from the old parameters
        SmallVector<Value *> newParam(CB->arg_begin(), CB->arg_end());

        /// Add the pipes we are propagating
        for (unsigned i = 0; i < ExtraTypes.size(); i++) {
          Argument *Arg =
              newCaller->getArg(i + newCaller->arg_size() - ExtraTypes.size());
          newParam.push_back(Arg);
        }

        Builder.SetInsertPoint(CBInNewFunc);
        /// Build the new call
        auto *newCB = Builder.CreateCall(newCallee, newParam);
        /// replace the old
        CBInNewFunc->replaceAllUsesWith(newCB);
        /// delete the old
        CBInNewFunc->eraseFromParent();
      }

      ToDelete.push_back(Func);
    }

    void handlePipeCreations() {
      if (PipeCreations.size() < 1)
        return;

      std::sort(PipeCreations.begin(), PipeCreations.end(),
                [](auto lhs, auto rhs) { return lhs.Func < rhs.Func; });
      llvm::SmallVector<CallBase *> collection;
      llvm::Function *cur_kernel = PipeCreations[0].Func;
      for (auto &elem : PipeCreations) {
        if (elem.Func != cur_kernel) {
          handlePipeCreations(cur_kernel, collection);
          cur_kernel = elem.Func;
          collection.clear();
        }
        collection.push_back(elem.CB);
      }
      handlePipeCreations(cur_kernel, collection);

      /// This will not delete the old call graph but make it such that it
      /// can be deleted by global DCE
      for (auto *F : ToDelete)
        sycl::removeKernelFuncAnnotation(F);
    }
  };

  void processLocalAnnotations() {
    LocalAnnotationVisitor LAV{*this};
    LAV.visit(M);
    if (AfterO3)
      LAV.handlePipeCreations();
  }

  bool run() {
    processLocalAnnotations();
    GlobalVariable *Annots = M.getGlobalVariable("llvm.global.annotations");
    if (Annots) {
      auto *Array = cast<ConstantArray>(Annots->getOperand(0));
      for (auto *E : Array->operand_values())
        processGlobalAnnotation(E);
    }
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

PreservedAnalyses LowerSYCLMetaDataPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  LSMDState(M).run();
  return PreservedAnalyses::none();
}

namespace llvm {
void initializeLowerSYCLMetaDataPass(PassRegistry &Registry);
} // namespace llvm

INITIALIZE_PASS(LowerSYCLMetaData, "lower-sycl-metadata",
                "prepare SYCL device code to optimizations", false, false)
ModulePass *llvm::createLowerSYCLMetaDataPass() {
  return new LowerSYCLMetaData();
}

char LowerSYCLMetaData::ID = 0;
