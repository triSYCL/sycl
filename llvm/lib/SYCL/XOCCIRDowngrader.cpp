//===- XOCCIRDowngrader.cpp                                      ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Erases and modifies IR incompatabilities with XOCC backend
//
// ===---------------------------------------------------------------------===//

#include <cstddef>
#include <regex>
#include <string>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/SYCL/XOCCIRDowngrader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/../../lib/IR/LLVMContextImpl.h"

using namespace llvm;

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// As a rule of thumb, if a pass to downgrade a part of the IR is added it
/// should have the LLVM version and date of patch/patch (if possible) that it
/// was added in so it can eventually be removed as xocc catches up
struct XOCCIRDowngrader : public ModulePass {

  static char ID; // Pass identification, replacement for typeid

  XOCCIRDowngrader() : ModulePass(ID) {}

  /// Removes byval bitcode function parameter attribute that is applied to
  /// pointer arguments of functions to state that they should technically be
  /// passed by value.
  ///
  /// While this attribute has been long standing and has been around since
  /// Clang/LLVM 7 it's diverged in the patch: D62319 which is the tail end of
  /// Clang/LLVM 9.
  ///
  /// This LLVM patch primarily added byval <Type> syntax in the IR. And a new
  /// attribute type, which is not to be confused with anything related to the
  /// type system. It basically just states the attribute is carrying around
  /// some Type/Value information rather than string/int/enum and gives it
  /// somewhere to store it.
  ///
  /// However now the bitcode writer will write out some different bitcode for
  /// byval that is unreadable by xocc and the bitcode reader upgrades any byval
  /// attributes to the new byval <Type> syntax. The way we currently work
  /// around this is:
  /// 1) Delete the ByVal attribute, erasing any byval <Type> syntax in the IR
  /// and then re-adding the old byval with no <Type> Syntax.
  /// 2) In the SYCL xocc script when we run this pass with opt we make sure we
  /// emit LLVM IR assembly language format and not bitcode format after this
  /// pass so that it doesn't run through the bitcode writer.
  /// 3) Inside the script we run it through an earlier LLVM assembler to create
  /// our bitcode, in this case the one packaged with xocc as it's easily
  /// accessible and we don't wish to carry around our own precompiled
  /// assemblers for every architecture in existence. Plus the bitcode it
  /// generates will always be bitcode compatible with xocc..
  /// 4) Feed it to xocc and be happy that it can consume it.
  ///
  /// The downside to this rather painful hack is that we're tied to the LLVM
  /// assembler that's packaged with xocc, which means newer LLVM IR may not be
  /// compatible with it. Which may lead to more problems in the future, one
  /// currently example of this problem is the renameBasicBlocks function inside
  /// this pass that renames all blocks to make the assembler happy. Without it
  /// the xocc llvm-as will die.
  ///
  /// An alternative to this would be to make our own alterations to the Bitcode
  /// writer so that it will optionally output the old bitcode encoding style of
  /// byval so that xocc can still read it in.
  ///
  /// Sadly we cannot just remove the byval attribute and ignore all the hacky
  /// workaround that come with keeping it as even though xocc will allow it to
  /// compile the resulting binary is not compatible with the XRT runtime. It
  /// will kill its execution when the kernel is launched, even for simple
  /// kernels.
  void resetByVal(Module &M) {
    for (auto &F : M.functions()) {
      for (auto &P : F.args()) {
         if (P.hasAttribute(llvm::Attribute::ByVal)) {
             P.removeAttr(llvm::Attribute::ByVal);
             P.addAttr(Attribute::get(M.getContext(), llvm::Attribute::ByVal,
                                      nullptr));
         }
      }

      // These appear on Call/Invoke Instructions as well
      for (auto &I : instructions(F))
        if (CallBase *CB = dyn_cast<CallBase>(&I)) {
          for (unsigned int i = 0; i < CB->getNumArgOperands(); ++i) {
            if (CB->paramHasAttr(i, llvm::Attribute::ByVal)) {
              CB->removeParamAttr(i, llvm::Attribute::ByVal);
              CB->addParamAttr(i,
                               Attribute::get(M.getContext(),
                                              llvm::Attribute::ByVal, nullptr));
            }
          }
        }
    }
  }

  /// This is part of the resetByVal work around, as we're using the xocc
  /// assember to assemble our code to bitcode we have to rename all the basic
  /// blocks. As the LLVM IR numbering of blocks seems to be a little too new
  /// for xocc's tastes.
  void renameBasicBlocks(Module &M) {
    int count;
    for (auto &F : M.functions()) {
        count = 0;
        for (auto &B : F)
          B.setName("label_" + Twine{count++});
    }
  }

  /// Removes nofree bitcode function attribute that is applied to
  /// functions to indicate that they do not deallocate memory.
  /// It was added in LLVM-9 (D49165), so as xocc catches up it can be removed
  /// Removes immarg (immutable arg) bitcode attribute that is applied to
  /// function parameters. It was added in LLVM-9 (D57825), so as xocc catches
  /// up it can be removed
  /// Removes WillReturn LLVM bitcode attribute from llvm/Doc/LangRef:
  ///
  /// "This function attribute indicates that a call of this function will
  ///  either exhibit undefined behavior or comes back and continues execution
  ///  at a point in the existing call stack that includes the current
  ///  invocation.
  ///  Annotated functions may still raise an exception, i.a., ``nounwind``
  ///  is not implied.
  ///  If an invocation of an annotated function does not return control back
  ///  to a point in the call stack, the behavior is undefined."
  ///
  /// Added in LLVM-10: rL364555 + D62801, this removal can be reverted as the
  /// xocc backend catches up. It seems unlikely removal will cause any problems
  /// as it appears to be an attribute that helps carry information to
  /// backends/other passes for further transformations.
  void removeAttributes(Module &M, ArrayRef<Attribute::AttrKind> Kinds) {
    for (auto &F : M.functions())
      for (auto Kind : Kinds) {
        F.removeAttribute(AttributeList::FunctionIndex, Kind);
        F.removeAttribute(AttributeList::ReturnIndex, Kind);
        for (auto &P : F.args())
          P.removeAttr(Kind);
        for (User *U : F.users())
          if (CallBase *CB = dyn_cast<CallBase>(U)) {
            CB->removeAttribute(AttributeList::FunctionIndex, Kind);
            CB->removeAttribute(AttributeList::ReturnIndex, Kind);
            for (unsigned int i = 0; i < CB->getNumArgOperands(); ++i) {
              CB->removeParamAttr(i, Kind);
            }
          }
      }
  }

  /// Remove Freeze instruction because xocc can't deal with them.
  /// This is not a safe transformation but since llvm survived with bugs cause
  /// by absence of freeze for many years, so i guess its its good enough for a
  /// prototype
  void removeFreezeInst(Module &M) {
    SmallVector<Instruction*, 16> ToRemove;
    for (auto& F : M.functions())
      for (auto& I : instructions(F))
        if (auto* Freeze = dyn_cast<FreezeInst>(&I)) {
          Freeze->replaceAllUsesWith(Freeze->getOperand(0));
          ToRemove.push_back(Freeze);
        }
    for (auto* I : ToRemove)
      I->eraseFromParent();
  }

  void removeFNegInst(Module& M) {
    SmallVector<Instruction*, 16> ToRemove;
    for (auto &F : M.functions())
      for (auto &I : instructions(F))
        if (auto *U = dyn_cast<UnaryOperator>(&I))
          if (U->getOpcode() == Instruction::FNeg) {
            Instruction* Sub = BinaryOperator::Create(BinaryOperator::FSub,
                                   ConstantFP::getZeroValueForNegation(
                                       U->getOperand(0)->getType()),
                                   U->getOperand(0));
            U->replaceAllUsesWith(Sub);
            Sub->insertBefore(U);
            ToRemove.push_back(U);
          }
    for (auto *I : ToRemove)
      I->eraseFromParent();
  }

  /// V++ has issues with intrinsic having different alignment attributes on
  /// inputs and outputs. So we remove alignment attributes.
  void removeMemIntrAlign(Module &M) {
    for (auto &F : M.functions())
      for (auto &I : instructions(F))
        if (auto *MI = dyn_cast<AnyMemIntrinsic>(&I))
          for (Use &U : MI->args())
            MI->removeAttribute(U.getOperandNo(),
                                Attribute::AttrKind::Alignment);
  }

  void lowerIntrinsic(Module &M) {
    IRBuilder<> B(M.getContext());
    SmallVector<Instruction *, 16> ToRemove;
    for (auto &F : M.functions())
      for (auto &I : instructions(F))
        if (auto *CI = dyn_cast<CallBase>(&I)) {
          if (CI->getIntrinsicID() == Intrinsic::abs) {
            B.SetInsertPoint(CI->getNextNode());
            Value *Cmp = B.CreateICmpSLT(
                CI->getArgOperand(0),
                ConstantInt::getNullValue(CI->getArgOperand(0)->getType()));
            Value *Sub = B.CreateSub(
                CI->getArgOperand(0),
                ConstantInt::getNullValue(CI->getArgOperand(0)->getType()));
            Value *ABS = B.CreateSelect(Cmp, Sub, CI->getArgOperand(0));
            CI->replaceAllUsesWith(ABS);
            ToRemove.push_back(CI);
          }
        }
    for (auto *I : ToRemove)
      I->eraseFromParent();
  }

  void convertPoinsonToZero(Module &M) {
    for (auto &PV : M.getContext().pImpl->PVConstants)
      PV.second.get()->replaceAllUsesWith(
          Constant::getNullValue(PV.second.get()->getType()));
  }

  void removeMetaDataValues(Module &M) {
    SmallVector<Instruction *, 16> ToDelete;
    for (auto &F : M.functions()) {
      if (llvm::none_of(F.args(), [&](Argument &A) {
            return A.getType()->isMetadataTy();
          }))
        continue;
      for (auto &U : F.uses()) {
        CallBase *CB = cast<CallBase>(U.getUser());
        assert(cast<FunctionType>(CB->getCalledFunction()->getType()->getPointerElementType())
                   ->getReturnType()
                   ->isVoidTy());
        ToDelete.push_back(CB);
      }
    }
    for (auto *I : ToDelete)
      I->eraseFromParent();
  }

  bool runOnModule(Module &M) override {
    resetByVal(M);
    removeAttributes(M, {Attribute::WillReturn, Attribute::NoFree,
                         Attribute::ImmArg, Attribute::NoSync,
                         Attribute::MustProgress, Attribute::NoUndef});
    renameBasicBlocks(M);
    removeFreezeInst(M);
    removeFNegInst(M);
    removeMemIntrAlign(M);

    lowerIntrinsic(M);
    removeMetaDataValues(M);

    convertPoinsonToZero(M);
    // The module probably changed
    return true;
  }
};

}

namespace llvm {
void initializeXOCCIRDowngraderPass(PassRegistry &Registry);
}

INITIALIZE_PASS(XOCCIRDowngrader, "xoccIRDowngrader",
  "pass that downgrades modern LLVM IR to something compatible with current xocc"
  "backend LLVM IR", false, false)
ModulePass *llvm::createXOCCIRDowngraderPass() {return new XOCCIRDowngrader();}

char XOCCIRDowngrader::ID = 0;
