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
#include "llvm/ADT/StringMap.h"
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

using namespace llvm;

namespace {
cl::opt<bool> AfterO3("lower-delayed-sycl-metadata", cl::Hidden,
                      cl::init(false));

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

  void lowerAsSideEffect(llvm::Value *V, StringRef XclId) {
    auto *F = dyn_cast<Function>(V);
    if (!F)
      return;

    HasChanged = true;
    Function *SideEffect = Intrinsic::getDeclaration(&M, Intrinsic::sideeffect);
    std::vector<Value *> Args;
    for (auto &A : F->args())
      Args.push_back(&A);
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

  /// Add a function with the vitis_kernel annotation attribute as an HLS kernel
  void markKernel(llvm::ConstantStruct *CS) {
    auto *F = cast<Function>(getUnderlyingObject(CS->getAggregateElement(0u)));
    F->addFnAttr("fpga.top.func", F->getName());
    F->addFnAttr("fpga.demangled.name", F->getName());
    F->setCallingConv(CallingConv::C);
    F->setLinkage(llvm::GlobalValue::ExternalLinkage);
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
      markKernel(CS);
    } else if (!AfterO3) { // Annotations that should be lowered before -O3
      if (AnnotKind == kindOf("xilinx_pipeline")) {
        lowerPipelineDecoration(CS);
      } else if (AnnotKind == kindOf("xilinx_partition_array")) {
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
    auto* Payload = cast<ConstantStruct>(PayloadCst);
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

  struct LocalAnnotationVisitor
      : public llvm::InstVisitor<LocalAnnotationVisitor> {
    LSMDState &Parent;
    using pipe_creation_rec_t = std::pair<llvm::Function *, llvm::CallBase *>;
    llvm::SmallVector<pipe_creation_rec_t> PipeCreations;

    LocalAnnotationVisitor(LSMDState &P) : Parent(P) {}

    void visitCallBase(CallBase &CB) {
      // Search for var_annotation having payload
      if (CB.getIntrinsicID() != Intrinsic::var_annotation) {
        auto unmangled_name =
            llvm::demangle(CB.getCalledFunction()->getName().str());
        if (unmangled_name.find(" CreatePipeFromPipeStorage_") !=
            std::string::npos) {
          // Parent.handlePipeCreation(CB);
          PipeCreations.emplace_back(CB.getFunction(), &CB);
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

    void forwardPipeCreation(llvm::Function *Kernel,
                             const size_t OriginalArgSize,
                             llvm::StringMap<std::size_t> &NameToArgOrder,
                             llvm::ArrayRef<llvm::StructType *> WrapperStruct) {
      llvm::SmallVector<std::pair<llvm::Instruction *, llvm::Instruction *>>
          ReplacementList;
      auto *Zero = llvm::ConstantInt::get(
          llvm::IntegerType::get(Kernel->getContext(), 32), 0);
      for (auto &BB : *Kernel) {
        for (auto &I : BB) {
          auto Rec = NameToArgOrder.find(I.getName());
          if (Rec != NameToArgOrder.end()) {
            auto Offset = Rec->second;
            auto *ULType = WrapperStruct[Offset]->getElementType(0);
            auto *AssociatedArg = Kernel->getArg(Offset);
            llvm::SmallVector<Value *, 2> Zeros{Zero, Zero};
            auto *ReplacementInst =
                llvm::GetElementPtrInst::Create(ULType, AssociatedArg, Zeros);
            for (auto *IUser : I.users()) {
              auto *CB = dyn_cast<CallBase>(IUser);
              if (CB) {
                auto CalledName =
                    llvm::demangle(CB->getCalledFunction()->getName().str());
                llvm::Instruction *NewInst = nullptr;
                if (CalledName.find("WritePipeBlockingINTEL") !=
                    std::string::npos) {
                  llvm::SmallVector<llvm::Type *, 2> WriteArgTypes{
                      ULType, ReplacementInst->getType()};
                  auto *WriteType =
                      llvm::FunctionType::get(nullptr, WriteArgTypes, false);
                  auto FifoWrite = llvm::Function::Create(
                      WriteType, GlobalValue::LinkageTypes::CommonLinkage,
                      llvm::formatv("llvm.fpga.fifo.push.{0}.{1}",
                                    Kernel->getName(), Offset));
                  auto *WrittenValPtr = CB->getArgOperand(1);

                  auto*WrittenVal = new llvm::LoadInst(ULType, WrittenValPtr, llvm::formatv("deref_pipe_val.{0}", Offset), CB);
                  llvm::SmallVector<Value*, 2> Args{WrittenVal, ReplacementInst};
                  NewInst = llvm::CallInst::Create(WriteType, FifoWrite, Args);
                } else if (CalledName.find("ReadPipeBlockingINTEL") !=
                           std::string::npos) {
                    llvm::SmallVector<llvm::Type *, 1> ReadArgTypes{ReplacementInst->getType()};
                    auto *ReadType =
                      llvm::FunctionType::get(ULType, ReadArgTypes, false);
                    auto FifoRead = llvm::Function::Create(
                      ReadType, GlobalValue::LinkageTypes::CommonLinkage,
                      llvm::formatv("llvm.fpga.fifo.pop.{0}.{1}",
                                    Kernel->getName(), Offset));
                    auto *StoredValPtr = CB->getArgOperand(1);
                    /*llvm::SmallVector<Value*, 2> Args{WrittenVal, ReplacementInst};
                    NewInst = llvm::CallInst::Create(ReadType, FifoRead, Args);*/
                }
              }
            }
          }
        }
      }
    }

    void handlePipeCreations(llvm::Function *Kernel,
                             llvm::ArrayRef<CallBase *> Creations) {
      const auto KernelArgSize = Kernel->arg_size();
      llvm::SmallVector<llvm::Type *> ExtraTypes;
      llvm::SmallVector<llvm::StructType *> WrapperStruct;
      llvm::StringMap<std::size_t> NameToOrder;
      size_t CurOffset = 0;
      for (auto *CB : Creations) {
        // Find usage to know the type of the object stored in the pipe
        assert(!CB->user_empty());
        auto *Usage = *(CB->user_begin());
        CallBase *UsageCb = dyn_cast<CallBase>(Usage);
        assert(UsageCb != nullptr);
        auto *PipeOp = UsageCb->getArgOperand(1);
        // assert(pipe_type->isPointerTy());
        PipeOp->dump();
        llvm::Type *PipeType = nullptr;
        if (auto *Alloca = dyn_cast<AllocaInst>(PipeOp)) {
          PipeType = Alloca->getAllocatedType();
        }
        assert(PipeType != nullptr);
        PipeType->dump();
        ExtraTypes.push_back(PipeType);
        NameToOrder.insert({CB->getName(), CurOffset++});
      }

      llvm::FunctionType *KernelType = Kernel->getFunctionType();
      std::vector<llvm::Type *> KernelArgs{};
      for (auto *Type : KernelType->params()) {
        KernelArgs.push_back(Type);
      }
      for (auto *Type : ExtraTypes) {
        auto *PipeStruct = llvm::StructType::create({&Type, 1});
        auto *PtrType = llvm::PointerType::get(PipeStruct, 0);
        KernelArgs.push_back(PtrType);
      }
      auto *NewKernelType =
          FunctionType::get(KernelType->getReturnType(), KernelArgs, false);
      auto KernelName = Kernel->getName().str();
      std::cerr << KernelName << std::endl;
#ifndef NDEBUG
      // As of now we do not handle non top-level kernels
      for (auto *UserPtr : Kernel->users()) {
        auto *Val = dyn_cast<CallBase>(UserPtr);
        assert(!Val);
      }
#endif
      Kernel->setName("to_be_replaced_kernel");
      auto *NewKernel = Function::Create(NewKernelType, Kernel->getLinkage(),
                                         KernelName, Kernel->getParent());
      llvm::ValueToValueMapTy Maptype;
      for (size_t i = 0 ; i < Kernel->arg_size() ; ++i) {
        Maptype.insert({Kernel->getArg(i), NewKernel->getArg(i)});  
      }
      llvm::SmallVector<llvm::ReturnInst *> RetInst{};
      llvm::CloneFunctionInto(NewKernel, Kernel, Maptype,
                              CloneFunctionChangeType::GlobalChanges, RetInst);
      // kernel->replaceAllUsesWith(new_kernel);
      Kernel->eraseFromParent();
      forwardPipeCreation(NewKernel, KernelArgSize, NameToOrder, WrapperStruct);
      // Perform annotation
    }

    void handlePipeCreations() {
      if (PipeCreations.size() < 1)
        return;

      std::sort(PipeCreations.begin(), PipeCreations.end());
      using pipe_creat_coll = llvm::SmallVector<CallBase *>;
      pipe_creat_coll collection{};
      llvm::Function *cur_kernel = PipeCreations[0].first;
      for (auto &elem : PipeCreations) {
        if (elem.first == cur_kernel) {
          collection.push_back(elem.second);
        } else {
          handlePipeCreations(cur_kernel, collection);
          cur_kernel = elem.first;
          collection = pipe_creat_coll{elem.second};
        }
      }
      handlePipeCreations(cur_kernel, collection);
    }
  };

  void processLocalAnnotations() {
    LocalAnnotationVisitor LAV{*this};
    LAV.visit(M);
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
