//===- KernelPropGen.cpp ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Retrieves the names of the kernels inside of the passed in file and places
// them into a text file. Possible to merge this into another pass if
// required, as it's a fairly trivial pass on its own.
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/SYCL/KernelPropGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> KernelPropGenOutput("sycl-kernel-propgen-output",
                                                cl::ReallyHidden);

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Enum for Address Space values, used in ASFixer pass and LLVM-SPIRV
enum SPIRAddressSpace {
  SPIRAS_Private,  // Address space: 0
  SPIRAS_Global,   // Address space: 1
  SPIRAS_Constant, // Address space: 2
  SPIRAS_Local,    // Address space: 3
  SPIRAS_Generic,  // Address space: 4
};

/// Retrieve the names for all kernels in the module and place them into a file
struct KernelPropGen : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  llvm::SmallDenseMap<llvm::AllocaInst *, unsigned, 8> UserSpecifiedDDRBanks;
  llvm::SmallDenseMap<llvm::Function *, std::string, 8> ExtraArgsMap;

  KernelPropGen() : ModulePass(ID) {}

  /// Test if a function is a SPIR kernel
  bool isKernel(const Function &F) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL ||
        F.hasFnAttribute("fpga.top.func"))
      return true;
    return false;
  }

  int GetWriteStreamID(StringRef Path) {
    int FileFD = 0;
    std::error_code EC = llvm::sys::fs::openFileForWrite(Path, FileFD);
    if (EC) {
      llvm::errs() << "Error in KernelPropGen Pass: " << EC.message() << "\n";
    }

    return FileFD;
  }

  static StringRef KindOf(const char *Str) {
    return StringRef(Str, strlen(Str) + 1);
  }

  void CollectUserSpecifiedDDRBanks(Function &F) {
    for (Instruction &I : instructions(F)) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || CB->getIntrinsicID() != Intrinsic::var_annotation)
        continue;
      auto *Alloca =
          dyn_cast_or_null<AllocaInst>(getUnderlyingObject(CB->getOperand(0)));
      auto *Str = cast<ConstantDataArray>(
          cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(1)))
              ->getOperand(0));
      if (!Alloca)
        continue;
      if (Str->getRawDataValues() != KindOf("xilinx_ddr_bank"))
        continue;
      Constant *Args =
          (cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(4)))
               ->getInitializer());
      unsigned Bank;
      if (auto *ZeroData = dyn_cast<ConstantAggregateZero>(Args))
        Bank = 0;
      else
        Bank = cast<ConstantInt>(Args->getOperand(0))->getZExtValue();

      UserSpecifiedDDRBanks[Alloca] = Bank;
    }
  }

  unsigned findDDRBankFor(Argument *Arg) {
    for (User *U : Arg->users()) {
      if (auto *Store = dyn_cast<StoreInst>(U))
        if (Store->getValueOperand() == Arg) {
          auto Lookup = UserSpecifiedDDRBanks.find(dyn_cast_or_null<AllocaInst>(
              getUnderlyingObject(Store->getPointerOperand())));
          if (Lookup == UserSpecifiedDDRBanks.end())
            continue;
          return Lookup->second;
        }
    }
    return 0;
  }

  /// Add the provided string as argument to all kernel functions that can reach
  /// the original function.
  void AddExtraArgsToCallers(Function *Original, std::string Additional) {
    SmallVector<llvm::Function *, 8> Stack;
    SmallPtrSet<llvm::Function *, 8> Visited;
    Stack.push_back(Original);
    while (!Stack.empty()) {
      llvm::Function *F = Stack.pop_back_val();
      if (!Visited.insert(F).second)
        continue;
      if (isKernel(*F)) {
        ExtraArgsMap[F] += Additional;
        continue;
      }
      for (User *U : F->users())
        if (auto *I = dyn_cast<Instruction>(U))
          Stack.push_back(I->getFunction());
    }
  }

  /// Find xilinx_kernel_param annotations, and record all provided arguments
  /// into ExtraArgsMap
  void CollectExtraArgs(Module &M) {
    SmallVector<User *, 8> Stack;
    for (GlobalVariable &V : M.globals()) {
      if (!isa<ConstantDataArray>(V.getInitializer()))
        continue;
      auto *Str = cast<ConstantDataArray>(V.getInitializer());
      if (Str->getRawDataValues() != KindOf("xilinx_kernel_param"))
        continue;
      Stack.clear();
      Stack.push_back(&V);
      while (!Stack.empty()) {
        User *U = Stack.pop_back_val();
        if (!isa<Instruction>(U)) {
          Stack.append(U->user_begin(), U->user_end());
          continue;
        }
        if (auto *CB = dyn_cast<CallBase>(U))
          if (CB->getIntrinsicID() == Intrinsic::var_annotation) {
            Constant *Args =
                (cast<GlobalVariable>(getUnderlyingObject(CB->getOperand(4)))
                     ->getInitializer());
            if (auto *C = dyn_cast<ConstantStruct>(Args)) {
              std::string ArgsStr;
              for (auto *V : C->operand_values()) {
                GlobalVariable *GV =
                    cast<GlobalVariable>(getUnderlyingObject(V));
                ArgsStr += cast<ConstantDataArray>(GV->getInitializer())
                               ->getRawDataValues()
                               .str() +
                           ' ';
              }
              AddExtraArgsToCallers(CB->getFunction(), ArgsStr);
            }
          }
      }
    }
  }

  void GenerateVPPPropertyFile(Module &M, llvm::raw_fd_ostream &O) {
    CollectExtraArgs(M);
    json::OStream J(O, 2);
    llvm::json::Array kernels{};

    J.objectBegin();
    J.attributeBegin("kernels");
    J.arrayBegin();
    for (auto &F : M.functions()) {
      if (isKernel(F)) {
        CollectUserSpecifiedDDRBanks(F);
        J.objectBegin();
        J.attribute("name", F.getName());
        J.attribute("extra_args", ExtraArgsMap[&F]);
        J.attributeBegin("memory_assignment");
        J.arrayBegin();
        for (auto &Arg : F.args()) {
          if (Arg.getType()->isPointerTy())
            // if the argument is a pointer in the global or constant
            // address space it should be assigned to an explicit default DDR
            // Bank of 0 to prevent assignment to DDR banks that are not 0.
            // This is to prevent mismatches between the SYCL runtime when
            // declaring OpenCL buffers and the pre-compiled kernel, XRT will
            // error out if there is a mismatch. Only OpenCL global memory is
            // assigned to a DDR bank, this includes constant as it's just
            // read-only global memory.
            // \todo When adding an explicit way for users to specify DDR banks
            // from the SYCL runtime this should be modified as well as the
            // buffer XRT extensions.
            if (Arg.getType()->isPointerTy() &&
                (Arg.getType()->getPointerAddressSpace() == SPIRAS_Global ||
                 Arg.getType()->getPointerAddressSpace() == SPIRAS_Constant)) {
              // This currently forces a default assignment of DDR banks to 0
              // as some platforms have different Default DDR banks and buffers
              // default to DDR Bank 0. Perhaps it is possible to query the
              // specific platform and reassign the buffers to different default
              // DDR banks based on the platform. But this would require a
              // change for every new platform. In either case, this puts in
              // infrastructure to assign DDR banks at compile time for a CU
              // if the information is passed down.
              // This: Assigns a Default 0 DDR bank to all initial compute
              // unit's, the _1 post-fix to the kernel name represents the
              // default compute unit name. If more than one CU is generated
              // (which we don't support yet in any case) then they would be
              // KernelName_2..KernelName_3 etc.
              J.objectBegin();
              J.attribute("arg_name", Arg.getName());
              J.attribute("bank_id", std::to_string(findDDRBankFor(&Arg)));
              J.objectEnd();
            }
        }
        J.arrayEnd();
        J.attributeEnd();
        J.objectEnd();
      }
    }
    J.arrayEnd();
    J.attributeEnd();
    J.objectEnd();
  }

  /// Visit all the functions of the module
  bool runOnModule(Module &M) override {
    llvm::raw_fd_ostream O(GetWriteStreamID(KernelPropGenOutput),
                           true /*close in destructor*/);

    if (O.has_error())
      return false;

    GenerateVPPPropertyFile(M, O);

    // The module probably changed
    return true;
  }
};

} // namespace

namespace llvm {
void initializeKernelPropGenPass(PassRegistry &Registry);
}

INITIALIZE_PASS(KernelPropGen, "kernelPropGen",
                "pass that finds kernel names and places them into a text file",
                false, false)
ModulePass *llvm::createKernelPropGenPass() { return new KernelPropGen(); }

char KernelPropGen::ID = 0;
