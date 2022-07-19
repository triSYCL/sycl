//===- ArchGenMLIR.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hydbrid(LLVMIR, MLIR) IR-gen used for the compiler implementation based of
// archgenlib
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#define DEBUG_TYPE "archgen-mlir"

namespace {

class ArchGenMLIRConsumer : public clang::ASTConsumer {
  clang::CompilerInstance &Instance;
  std::unique_ptr<clang::ASTConsumer> LLVMIRASTConsumer;
  llvm::Module *LLVMModule;
  llvm::LLVMContext *LLVMCtx;
  mlir::MLIRContext ctx;

  void InitMLIR() {
    /// Make sure that MLIR libs are properly linked with
    ctx.disableMultithreading();
    ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  }

public:
  ArchGenMLIRConsumer(clang::CompilerInstance &Instance,
                      std::unique_ptr<clang::ASTConsumer> Consumer,
                      llvm::Module *Module)
      : Instance(Instance), LLVMIRASTConsumer(std::move(Consumer)),
        LLVMModule(Module), LLVMCtx(&Module->getContext()) {
    InitMLIR();
  }

  // Below is the exposed API of clang::ASTConsumer most place just need to
  // forward to LLVMIRGen clang::ASTConsumer
  //===----------------------------------------------------------------------===//
  void Initialize(clang::ASTContext &Context) override {
    return LLVMIRASTConsumer->Initialize(Context);
  }
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    return LLVMIRASTConsumer->HandleCXXStaticMemberVarInstantiation(VD);
  }
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
    llvm::errs() << "---------------------------\n";
    for (auto Elem : D)
      Elem->dumpColor();
    return LLVMIRASTConsumer->HandleTopLevelDecl(D);
  }
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override {
    return LLVMIRASTConsumer->HandleInlineFunctionDefinition(D);
  }
  void HandleInterestingDecl(clang::DeclGroupRef D) override {
    return LLVMIRASTConsumer->HandleInterestingDecl(D);
  }
  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    return LLVMIRASTConsumer->HandleTranslationUnit(Ctx);
  }
  void HandleTagDeclDefinition(clang::TagDecl *D) override {
    return LLVMIRASTConsumer->HandleTagDeclDefinition(D);
  }
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override {
    return LLVMIRASTConsumer->HandleTagDeclRequiredDefinition(D);
  }
  void HandleCXXImplicitFunctionInstantiation(clang::FunctionDecl *D) override {
    return LLVMIRASTConsumer->HandleCXXImplicitFunctionInstantiation(D);
  }
  void HandleTopLevelDeclInObjCContainer(clang::DeclGroupRef D) override {
    return LLVMIRASTConsumer->HandleTopLevelDeclInObjCContainer(D);
  }
  void HandleImplicitImportDecl(clang::ImportDecl *D) override {
    return LLVMIRASTConsumer->HandleImplicitImportDecl(D);
  }
  void CompleteTentativeDefinition(clang::VarDecl *D) override {
    return LLVMIRASTConsumer->CompleteTentativeDefinition(D);
  }
  void CompleteExternalDeclaration(clang::VarDecl *D) override {
    return LLVMIRASTConsumer->CompleteExternalDeclaration(D);
  }
  void AssignInheritanceModel(clang::CXXRecordDecl *RD) override {
    return LLVMIRASTConsumer->AssignInheritanceModel(RD);
  }
  void HandleVTable(clang::CXXRecordDecl *RD) override {
    return LLVMIRASTConsumer->HandleVTable(RD);
  }
  clang::ASTMutationListener *GetASTMutationListener() override {
    return LLVMIRASTConsumer->GetASTMutationListener();
  }
  clang::ASTDeserializationListener *GetASTDeserializationListener() override {
    return LLVMIRASTConsumer->GetASTDeserializationListener();
  }
  void PrintStats() override { return LLVMIRASTConsumer->PrintStats(); }
  bool shouldSkipFunctionBody(clang::Decl *D) override {
    return LLVMIRASTConsumer->shouldSkipFunctionBody(D);
  }
};

class ArchGenMLIRAction : public clang::PluginASTAction {

  /// Original Main actions being replaced. it usually should be a
  /// clang::CodeGenAction. but not always
  std::unique_ptr<clang::FrontendAction> Inner;

public:
  clang::CodeGenAction *getInnerAsCodeGenAction() {
    /// clang is built without RTTI so dynamic_cast cannot be used to
    /// check if Inner is a clang::CodeGenAction. But the only
    /// clang::FrontendAction that returns true to hasIRSupport is
    /// clang::CodeGenAction so we used this to check
    if (!Inner->hasIRSupport())
      return nullptr;
    return static_cast<clang::CodeGenAction *>(Inner.get());
  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override {
    auto InnerConsumer = Inner->CreateASTConsumer(CI, InFile);

    /// clang::CodeGenAction should be wrapped but not other kinds of main
    /// actions
    if (auto *CGAct = getInnerAsCodeGenAction())
      return std::make_unique<ArchGenMLIRConsumer>(
          CI, std::move(InnerConsumer), CGAct->getCodeGenerator()->GetModule());
    return InnerConsumer;
  }

  /// Assignements will be set on our clang::FrontendAction and not on the Inner
  /// clang::FrontendAction So before This function copies our data to the Inner
  void copySelfToInner() {
    Inner->setCurrentInput(getCurrentInput(), takeCurrentASTUnit());
    Inner->setCompilerInstance(&getCompilerInstance());
  }

  bool hasIRSupport() const override { return Inner->hasIRSupport(); }

  void ExecuteAction() override {
    copySelfToInner();
    return Inner->ExecuteAction();
  }

  void EndSourceFileAction() override { return Inner->EndSourceFileAction(); }

  void takeMainActionToReplace(std::unique_ptr<FrontendAction> Old) override {
    Inner = std::move(Old);
  }

  clang::PluginASTAction::ActionType getActionType() override {
    return clang::PluginASTAction::ActionType::ReplaceAndReuseAction;
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

public:
  ArchGenMLIRAction() {}
};

} // namespace

static clang::FrontendPluginRegistry::Add<ArchGenMLIRAction>
    X("archgen-mlir", "generate approximations in mlir");
