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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"

#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "archgen/Aprox/Aprox.h"
#include "archgen/FixedPt/FixedPt.h"

#define DEBUG_TYPE "archgen-mlir"

namespace {

class MLIRGenState {
  void InitMLIR() {
    /// Make sure that MLIR libs are properly linked with
    ctx.disableMultithreading();
    ctx.loadDialect<mlir::LLVM::LLVMDialect, mlir::arith::ArithmeticDialect,
                    mlir::func::FuncDialect, archgen::aprox::AproxDialect,
                    archgen::fixedpt::FixedPtDialect>();

    builder.setInsertionPointToStart(module->getBody());
  }

public:
  mlir::MLIRContext ctx;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  clang::SourceManager &SM;
  clang::CodeGen::CodeGenModule CGM;
  llvm::DenseMap<clang::FunctionDecl *, mlir::func::FuncOp> FunctionMap;

  MLIRGenState(clang::CompilerInstance &CI, llvm::Module *LLVMModule)
      : builder(&ctx), loc(builder.getUnknownLoc()),
        module(mlir::ModuleOp::create(loc)), SM(CI.getSourceManager()),
        CGM(CI.getASTContext(),
            CI.getPreprocessor().getHeaderSearchInfo().getHeaderSearchOpts(),
            CI.getPreprocessor().getPreprocessorOpts(), CI.getCodeGenOpts(),
            *LLVMModule, CI.getPreprocessor().getDiagnostics()) {
    InitMLIR();
  }
  mlir::Location getMLIRLocation(clang::SourceLocation loc) {
    auto spellingLoc = SM.getSpellingLoc(loc);
    auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
    auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
    auto fileId = SM.getFilename(spellingLoc);

    auto ctx = module->getContext();
    return mlir::FileLineColLoc::get(ctx, fileId, lineNumber, colNumber);
  }

  mlir::Type getMLIRType(clang::QualType Cqtype) {
    auto *Ctype = Cqtype->getUnqualifiedDesugaredType();
    if (auto *BT = llvm::dyn_cast<clang::BuiltinType>(Ctype)) {
      if (Ctype->isBooleanType())
        return builder.getIntegerType(8);
      llvm::Type *T = CGM.getTypes().ConvertType(Cqtype);
      if (T->isVoidTy())
        return builder.getNoneType();
      if (T->isFloatTy())
        return builder.getF32Type();
      if (T->isDoubleTy())
        return builder.getF64Type();
      if (T->isX86_FP80Ty())
        return builder.getF80Type();
      if (T->isFP128Ty())
        return builder.getF128Type();
      if (auto IT = llvm::dyn_cast<llvm::IntegerType>(T))
        return builder.getIntegerType(IT->getBitWidth());
    }

    if (auto *FT = llvm::dyn_cast<clang::FunctionProtoType>(Ctype)) {
      llvm::SmallVector<mlir::Type> Args;
      llvm::transform(FT->param_types(), std::back_inserter(Args),
                      [&](clang::QualType Elem) { return getMLIRType(Elem); });
      return mlir::FunctionType::get(&ctx, Args,
                                     {getMLIRType(FT->getReturnType())});
    }

    Cqtype->dump();
    llvm_unreachable("unhandeled type");
  }

  mlir::func::FuncOp GetOrCreateFunc(clang::FunctionDecl *FD) {
    auto &MLIRFunc = FunctionMap[FD];
    if (MLIRFunc)
      return MLIRFunc;

    MLIRFunc = builder.create<mlir::func::FuncOp>(
        loc, CGM.getMangledName(FD).str(),
        getMLIRType(FD->getType()).cast<mlir::FunctionType>());
    return MLIRFunc;
  }
};

struct MLIREmiter : public clang::StmtVisitor<MLIREmiter, mlir::Value> {
  using Base = clang::StmtVisitor<MLIREmiter, mlir::Value>;
  MLIRGenState &state;
  MLIREmiter(MLIRGenState &s) : state(s) {}
  void Emit(clang::FunctionDecl* FD, mlir::func::FuncOp MLIRFunc) {
    mlir::Block* block = MLIRFunc.addEntryBlock();
    state.builder.setInsertionPointToStart(block);
    Visit(FD->getBody());
  }
  mlir::Value VisitCompoundStmt(clang::CompoundStmt* CS) {
    for (auto* Elem : CS->body()) {
      Visit(Elem);
    }
    return {};
  }
  mlir::Value VisitCallExpr(clang::CallExpr* CE) {
    CE->dumpColor();
    return {};
  }
  mlir::Value VisitReturnStmt(clang::ReturnStmt *RS) {
    mlir::Value val = Visit(RS->getRetValue());
    state.builder.create<mlir::func::ReturnOp>(
        state.getMLIRLocation(RS->getReturnLoc()), val.getType(), val);
    return {};
  }
  mlir::Value VisitExpr(clang::Stmt *S) {
    S->dump();
    llvm_unreachable("unhandled Stmt");
  }
};

class ArchGenMLIRConsumer : public clang::ASTConsumer {
  clang::CompilerInstance &CI;
  std::unique_ptr<clang::ASTConsumer> LLVMIRASTConsumer;
  llvm::Module *LLVMModule;
  llvm::LLVMContext *LLVMCtx;
  MLIRGenState state;
  MLIREmiter Visitor;

public:
  ArchGenMLIRConsumer(clang::CompilerInstance &CI,
                      std::unique_ptr<clang::ASTConsumer> Consumer,
                      llvm::Module *LLVMModule)
      : CI(CI), LLVMIRASTConsumer(std::move(Consumer)), LLVMModule(LLVMModule),
        LLVMCtx(&LLVMModule->getContext()), state(CI, LLVMModule),
        Visitor(state) {}

  std::deque<clang::FunctionDecl *> FuncToEmit;

  bool isOurAnnotationAttr(clang::AnnotateAttr *Attr) {
    return Attr->getAnnotation().startswith("archgen_mlir");
  }

  void addToEmit(clang::FunctionDecl* FD) {
    LLVM_DEBUG(FD->dump(llvm::dbgs()));
    state.GetOrCreateFunc(FD);
    FuncToEmit.push_back(FD);
  }

  void FillFunctionBody(clang::FunctionDecl* FD) {
    mlir::func::FuncOp MLIRFunc = state.GetOrCreateFunc(FD);
    Visitor.Emit(FD, MLIRFunc);
  }

  void runEmit() {
    for (auto* FD : FuncToEmit) {
      if (FD->getBody())
        FillFunctionBody(FD);
    }
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
    llvm::SmallVector<clang::Decl *> NewDG;
    NewDG.reserve(std::distance(D.begin(), D.end()));
    for (auto Elem : D) {
      if (auto *FD = llvm::dyn_cast<clang::FunctionDecl>(Elem))
        if (auto *AA = FD->getAttr<clang::AnnotateAttr>())
          if (isOurAnnotationAttr(AA)) {
            addToEmit(FD);
            continue;
          }
      NewDG.push_back(Elem);
    }
    return LLVMIRASTConsumer->HandleTopLevelDecl(D);
  }
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override {
    return LLVMIRASTConsumer->HandleInlineFunctionDefinition(D);
  }
  void HandleInterestingDecl(clang::DeclGroupRef D) override {
    return LLVMIRASTConsumer->HandleInterestingDecl(D);
  }
  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    runEmit();
    state.module->dump();
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
