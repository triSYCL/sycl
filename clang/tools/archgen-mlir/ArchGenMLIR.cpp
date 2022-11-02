//===- ArchGenMLIR.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hybrid(LLVMIR, MLIR) IR-gen used for the compiler implementation based of
// archgenlib
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/Sema.h"

#include "archgen/Approx/Approx.h"
#include "archgen/Approx/Passes.h"
#include "archgen/FixedPt/FixedPt.h"
#include "archgen/FixedPt/Passes.h"

using namespace clang;
using namespace archgen;
namespace cl = llvm::cl;
namespace func = mlir::func;
namespace arith = mlir::arith;
namespace memref = mlir::memref;
namespace LLVM = mlir::LLVM;

#define DEBUG_TYPE "archgen-mlir"
namespace {

cl::opt<std::string>
    MLIROutput("archgen-mlir-mlir-output", cl::init("-"), cl::Optional,
               cl::desc("MLIR output of the ArchGenMLIR plugin"));

cl::opt<bool>
    DontLinkMLIR("archgen-mlir-dont-link-mlir", cl::init(false), cl::Optional,
                 cl::desc("do not link the MLIR output into the LLVM IR"));

cl::opt<bool> StopAtApprox(
    "archgen-mlir-stop-at-approx", cl::init(false), cl::Optional,
    cl::desc("stop after emitting and cleanup the approx dialect"));

cl::opt<bool> ListNotEmit(
    "archgen-mlir-list-dont-emit", cl::init(false), cl::Optional,
    cl::desc("instead of emitting the MLIR list what would be emmitted"));

/// Output MLIR module to a file
mlir::LogicalResult outputMLIR(llvm::StringRef file, mlir::ModuleOp module) {
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(file, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }
  outputFile->keep();
  module->print(outputFile->os());
  return mlir::success();
}

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

enum class TypeLookupOpt {
  isTemplate = 1 << 0,
  isInDetailNamespace = 1 << 1,

  LLVM_MARK_AS_BITMASK_ENUM(isInDetailNamespace)
};

/// Utility to identify classes that need special handling like
/// ::archgenlib::detail::ToBeFolded
class Util {
  using DeclContextDesc = std::pair<clang::Decl::Kind, llvm::StringRef>;

  template <size_t N>
  static constexpr DeclContextDesc MakeDeclContextDesc(clang::Decl::Kind K,
                                                       const char (&Str)[N]) {
    return DeclContextDesc{K, llvm::StringRef{Str, N - 1}};
  }

  static constexpr DeclContextDesc MakeDeclContextDesc(clang::Decl::Kind K,
                                                       llvm::StringRef SR) {
    return DeclContextDesc{K, SR};
  }

  static bool matchContext(const clang::DeclContext *Ctx,
                           llvm::ArrayRef<Util::DeclContextDesc> Scopes) {
    // The idea: check the declaration context chain starting from the item
    // itself. At each step check the context is of expected kind
    // (namespace) and name.
    llvm::StringRef Name = "";

    for (const auto &Scope : llvm::reverse(Scopes)) {
      clang::Decl::Kind DK = Ctx->getDeclKind();
      if (DK != Scope.first)
        return false;

      switch (DK) {
      case clang::Decl::Kind::ClassTemplateSpecialization:
        // ClassTemplateSpecializationDecl inherits from CXXRecordDecl
      case clang::Decl::Kind::CXXRecord:
        Name = cast<clang::CXXRecordDecl>(Ctx)->getName();
        break;
      case clang::Decl::Kind::Namespace:
        Name = cast<clang::NamespaceDecl>(Ctx)->getName();
        break;
      default:
        llvm_unreachable("matchContext: decl kind not supported");
      }
      if (Name != Scope.second)
        return false;
      Ctx = Ctx->getParent();
    }
    return Ctx->isTranslationUnit() ||
           (Ctx->isExternCXXContext() &&
            Ctx->getEnclosingNamespaceContext()->isTranslationUnit());
  }

  static bool
  matchQualifiedTypeName(clang::QualType Ty,
                         llvm::ArrayRef<Util::DeclContextDesc> Scopes) {
    const clang::CXXRecordDecl *RecTy = Ty->getAsCXXRecordDecl();

    if (!RecTy)
      return false; // only classes/structs supported
    const auto *Ctx = cast<clang::DeclContext>(RecTy);
    return Util::matchContext(Ctx, Scopes);
  }

public:

  /// Top-level function that should be used
  static bool isArchGenType(clang::QualType Qt, llvm::StringRef Name,
                            TypeLookupOpt Opts) {
    clang::Decl::Kind ClassDeclKind =
        static_cast<bool>(Opts & TypeLookupOpt::isTemplate)
            ? clang::Decl::Kind::ClassTemplateSpecialization
            : clang::Decl::Kind::CXXRecord;
    if (static_cast<bool>(Opts & TypeLookupOpt::isInDetailNamespace))
      return Util::matchQualifiedTypeName(
          Qt,
          {Util::MakeDeclContextDesc(clang::Decl::Kind::Namespace,
                                     "archgenlib"),
           Util::MakeDeclContextDesc(clang::Decl::Kind::Namespace, "detail"),
           Util::MakeDeclContextDesc(ClassDeclKind, Name)});
    return Util::matchQualifiedTypeName(
        Qt,
        {Util::MakeDeclContextDesc(clang::Decl::Kind::Namespace, "archgenlib"),
         Util::MakeDeclContextDesc(ClassDeclKind, Name)});
  }
};

/// Utility to detect Annotations on Declarations
namespace Annot {

constexpr llvm::StringLiteral GenericOp = "archgen_mlir_generic_op";
constexpr llvm::StringLiteral GenAsMLIR = "archgen_mlir_emit_as_mlir";
constexpr llvm::StringLiteral TopLevel = "archgen_mlir_top_level";
constexpr llvm::StringLiteral KindConstant = "constant";
constexpr llvm::StringLiteral KindEvaluate = "evaluate";
constexpr llvm::StringLiteral KindVariable = "variable";
constexpr llvm::StringLiteral KindParam = "parameter";

bool hasAnnotation(clang::Decl *D, llvm::StringRef Annot) {
  return llvm::any_of(
      D->specific_attrs<clang::AnnotateAttr>(),
      [&](clang::AnnotateAttr *AA) { return AA->getAnnotation() == Annot; });
}

} // namespace Annot

/// Stores all the MLIR related constructs and provide basic utilities to emit
/// MLIR from clang AST
class MLIRGenState {
  void InitMLIR() {
    ctx.disableMultithreading();
    ctx.loadDialect<mlir::LLVM::LLVMDialect, arith::ArithmeticDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    approx::ApproxDialect, fixedpt::FixedPtDialect>();
    mlir::registerLLVMDialectTranslation(ctx);

    declBuilder.setInsertionPointToStart(module->getBody());
    builder.clearInsertionPoint();
  }

  /// List of functions that need to get emitted
  std::deque<clang::FunctionDecl *> FuncToEmit;
public:
  /// This class is used to easily pass around all the needed state to perform
  /// MLIR emission from clang AST so most data members are public

  mlir::MLIRContext ctx;
  mlir::OpBuilder builder;
  mlir::OpBuilder declBuilder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  clang::ASTContext &ASTctx;
  clang::SourceManager &SM;
  clang::CodeGen::CodeGenModule CGM;
  llvm::DenseMap<clang::FunctionDecl *, func::FuncOp> FunctionMap;
  llvm::DenseMap<clang::Decl *, mlir::Value> localDeclMap;

  MLIRGenState(clang::CompilerInstance &CI, llvm::Module *LLVMModule)
      : builder(&ctx), declBuilder(&ctx),
        module(mlir::ModuleOp::create(builder.getUnknownLoc())),
        ASTctx(CI.getASTContext()), SM(CI.getSourceManager()),
        CGM(CI.getASTContext(),
            CI.getPreprocessor().getHeaderSearchInfo().getHeaderSearchOpts(),
            CI.getPreprocessor().getPreprocessorOpts(), CI.getCodeGenOpts(),
            *LLVMModule, CI.getPreprocessor().getDiagnostics()) {
    InitMLIR();
  }

  /// This class should stay unique
  MLIRGenState(const MLIRGenState &) = delete;
  MLIRGenState &operator=(const MLIRGenState &) = delete;

  /// Convert clang::SourceLocation to mlir::SourceLocation
  mlir::Location getMLIRLocation(clang::SourceLocation loc) {
    auto spellingLoc = SM.getSpellingLoc(loc);
    auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
    auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
    auto fileId = SM.getFilename(spellingLoc);

    return mlir::FileLineColLoc::get(&ctx, fileId, lineNumber, colNumber);
  }

  const clang::TemplateArgumentList &getTemplateArgList(clang::QualType Ty) {
    return cast<clang::ClassTemplateSpecializationDecl>(
               Ty->getAsRecordDecl())
        ->getTemplateArgs();
  }

  mlir::Type getFixedPointType(clang::QualType Ty) {
    /// Assuming Ty is FixedNumber<FixedFormat<MSB, LSB, signed OR unsigned>>
    auto &TAL = getTemplateArgList(Ty);
    assert(TAL.size() == 1);

    auto &TAL2 = getTemplateArgList(TAL.get(0).getAsType());
    assert(TAL2.size() == 3);

    int msb = TAL2.get(0).getAsIntegral().getExtValue();
    int lsb = TAL2.get(1).getAsIntegral().getExtValue();
    bool is_signed = TAL2.get(2).getAsType()->isSignedIntegerType();

    return fixedpt::FixedPtType::get(&ctx, msb, lsb, is_signed);
  }

  /// Convert clang::QualType to mlir::Type
  mlir::Type getMLIRType(clang::QualType Cqtype) {
    /// If we want to handle more complex C++ types we will need to add caching
    auto Cqdesugar = Cqtype.getDesugaredType(ASTctx);

    /// Types we need to handle specially
    if (Util::isArchGenType(Cqdesugar, "ToBeFolded",
                            TypeLookupOpt::isInDetailNamespace))
      return approx::toBeFoldedType::get(&ctx);
    if (Util::isArchGenType(Cqdesugar, "FixedNumber",
                            TypeLookupOpt::isTemplate))
      return getFixedPointType(Cqdesugar);

    auto *Ctype = Cqdesugar->getUnqualifiedDesugaredType();
    if (auto *BT = dyn_cast<clang::BuiltinType>(Ctype)) {
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
      if (auto IT = dyn_cast<llvm::IntegerType>(T))
        return builder.getIntegerType(IT->getBitWidth());
    }

    if (auto *FT = dyn_cast<clang::FunctionProtoType>(Ctype)) {
      llvm::SmallVector<mlir::Type> Args;
      llvm::transform(FT->param_types(), std::back_inserter(Args),
                      [&](clang::QualType Elem) { return getMLIRType(Elem); });
      return mlir::FunctionType::get(&ctx, Args,
                                     {getMLIRType(FT->getReturnType())});
    }

    if (auto *PT = Ctype->getPointeeType().getTypePtr())
      return LLVM::LLVMPointerType::get(getMLIRType(clang::QualType(PT, 0)));

    Cqtype->dump();
    llvm::errs() << "canonicalized to:\n";
    Ctype->dump();
    llvm_unreachable("unhandled type");
  }

  /// Return a FuncOp for the provided Decl.
  /// FuncOp are created empty
  func::FuncOp GetOrCreateFunc(clang::FunctionDecl *FD) {
    auto &MLIRFunc = FunctionMap[FD];
    if (MLIRFunc)
      return MLIRFunc;

    LLVM_DEBUG({
      llvm::dbgs() << "add to emit";
      if (Annot::hasAnnotation(FD, Annot::TopLevel))
        llvm::dbgs() << " top-level";
      llvm::dbgs() << ":\n";
      FD->dump(llvm::dbgs());
    });

    /// If it is not in the map is it not going to get emitter on its own
    /// So we add it to the list of functions to emit
    FuncToEmit.push_back(FD);

    /// The top-level function need to stay, all the reset should be deleted by
    /// inlining
    bool isPrivate = !Annot::hasAnnotation(FD, Annot::TopLevel);
    mlir::StringAttr visibility =
        declBuilder.getStringAttr(isPrivate ? "private" : "public");

    /// Create the functions empty such that they can be referenced without
    /// being generated yet
    MLIRFunc = declBuilder.create<func::FuncOp>(
        getMLIRLocation(FD->getBeginLoc()), CGM.getMangledName(FD).str(),
        getMLIRType(FD->getType()).cast<mlir::FunctionType>(), visibility);
    return MLIRFunc;
  }
  std::deque<clang::FunctionDecl *> takeFuncToEmit() {
    return std::move(FuncToEmit);
  }
};

/// Recursively traverse clang::Stmt while emitting them
/// If the generator needed to emit more complex Statements like l-values it
/// should not return an mlir::Value but for now it is kept simple.
struct MLIREmitter : public clang::StmtVisitor<MLIREmitter, mlir::Value> {
  using Base = clang::StmtVisitor<MLIREmitter, mlir::Value>;
  MLIRGenState &state;
  MLIREmitter(MLIRGenState &s) : state(s) {}

  int64_t evaluateConstant(clang::Expr* E) {
    clang::Expr::EvalResult result;
    E->EvaluateAsRValue(result, state.ASTctx);
    assert(result.Val.hasValue());
    return result.Val.getInt().getZExtValue();
  }

  /// Entry point of the emitter
  void Emit(clang::FunctionDecl *FD, func::FuncOp MLIRFunc) {
    /// TODO: this should get split into EmitFunc and EmitBlock
    mlir::Block *block = MLIRFunc.addEntryBlock();
    state.builder.setInsertionPointToStart(block);

    state.localDeclMap.clear();
    for (unsigned idx = 0; idx < FD->getNumParams(); idx++) {
      Decl *PD = FD->getParamDecl(idx);
      mlir::Value BA = block->getArgument(idx);
      state.localDeclMap[PD] = BA;
    }

    Visit(FD->getBody());

    if (!block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (FD->getReturnType()->isVoidType()) {
        mlir::Value zero =
            state.builder
                .create<arith::ConstantIntOp>(
                    state.getMLIRLocation(FD->getBody()->getEndLoc()), 0,
                    MLIRFunc.getFunctionType().getResult(0))
                .getResult();
        state.builder.create<func::ReturnOp>(
            state.getMLIRLocation(FD->getBody()->getEndLoc()), zero);
      } else
        llvm_unreachable("all non void functions must have a return for now");
    }
  }

  mlir::Value VisitCompoundStmt(clang::CompoundStmt *CS) {
    for (auto *Elem : CS->body())
      Visit(Elem);
    return {};
  }

  mlir::Value VisitIntegerLiteral(clang::IntegerLiteral *IL) {
    mlir::Type MLIRType = state.getMLIRType(IL->getType());
    return state.builder
        .create<arith::ConstantIntOp>(
            state.getMLIRLocation(IL->getLocation()),
            IL->getValue().getZExtValue(), MLIRType)
        .getResult();
  }

  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *E) {
    return Visit(E->getReplacement());
  }

  mlir::Value VisitGenericOpCall(clang::CallExpr *CE) {
    assert(Annot::hasAnnotation(CE->getCalleeDecl(), Annot::GenericOp));
    llvm::StringRef Kind =
        cast<clang::StringLiteral>(CE->getArg(0)->IgnoreImplicit())
            ->getString();

    if (Kind == Annot::KindConstant) {
      llvm::FixedPointSemantics sema =
          state.getMLIRType(CE->getArg(1)->getType())
              .cast<fixedpt::FixedPtType>()
              .getFixedPointSemantics();
      auto *DRExpr = cast<clang::DeclRefExpr>(
          *cast<clang::CXXTemporaryObjectExpr>(CE->getArg(1))
               ->children()
               .begin());
      clang::APValue *Cvalue =
          cast<clang::VarDecl>(DRExpr->getDecl())->getEvaluatedValue();
      llvm::APFixedPoint FPvalue(Cvalue->getInt(), sema);
      auto attr =
          fixedpt::FixedPointAttr::get(&state.ctx, std::move(FPvalue));
      return state.builder
          .create<approx::ConstantOp>(state.getMLIRLocation(CE->getBeginLoc()),
                                      attr)
          ->getResults()[0];
    }
    if (Kind == Annot::KindEvaluate)
      return state.builder.create<approx::EvaluateOp>(
          state.getMLIRLocation(CE->getBeginLoc()),
          state.getMLIRType(CE->getType()), Visit(CE->getArg(1)),
          static_cast<archgen::approx::ApproxMode>(
              evaluateConstant(CE->getArg(2))));
    if (Kind == Annot::KindVariable)
      return state.builder.create<approx::VariableOp>(
          state.getMLIRLocation(CE->getBeginLoc()), Visit(CE->getArg(1)));
    if (Kind == Annot::KindParam)
      return state.builder.create<approx::ParameterOp>(
          state.getMLIRLocation(CE->getBeginLoc()),
          state.getMLIRType(CE->getType()), Visit(CE->getArg(1)));

    llvm::SmallVector<mlir::Value> Args;
    for (int64_t idx = 1; idx < CE->getNumArgs(); idx++)
      Args.push_back(Visit(CE->getArg(idx)));

    return state.builder
        .create<approx::GenericOp>(state.getMLIRLocation(CE->getBeginLoc()),
                                   Args, Kind)
        .output();
  }

  mlir::Value VisitCallExpr(clang::CallExpr *CE) {
    if (Annot::hasAnnotation(CE->getCalleeDecl(), Annot::GenericOp))
      return VisitGenericOpCall(CE);

    llvm::SmallVector<mlir::Value> Args;
    for (auto *E : CE->arguments())
      Args.push_back(Visit(E));

    func::FuncOp Callee =
        state.GetOrCreateFunc(cast<clang::FunctionDecl>(CE->getCalleeDecl()));

    return state.builder
        .create<func::CallOp>(state.getMLIRLocation(CE->getBeginLoc()), Callee,
                              Args)
        ->getResults()[0];
  }

  mlir::Value VisitReturnStmt(clang::ReturnStmt *RS) {
    mlir::Value val = Visit(RS->getRetValue());
    state.builder.create<func::ReturnOp>(
        state.getMLIRLocation(RS->getReturnLoc()), val);
    return {};
  }

  mlir::Value VisitDeclRefExpr(clang::DeclRefExpr *DR) {
    return state.localDeclMap[DR->getDecl()];
  }

  mlir::Value VisitExprWithCleanups(clang::ExprWithCleanups *EWC) {
    assert(!EWC->cleanupsHaveSideEffects());
    return Visit(EWC->getSubExpr());
  }

  mlir::Value VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *OCE) {
    assert(OCE->isAssignmentOp());
    mlir::Value ptr = Visit(OCE->getArg(0));
    mlir::Value val = Visit(OCE->getArg(1));
    state.builder.create<LLVM::StoreOp>(
        state.getMLIRLocation(OCE->getExprLoc()), val, ptr);
    return ptr;
  }

  mlir::Value
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *MTE) {
    return Visit(MTE->getSubExpr());
  }

  /// All unhandled clang::Stmt come here so we emit an error
  mlir::Value VisitStmt(clang::Stmt *S) {
    S->dump();
    llvm_unreachable("unhandled Stmt");
  }
};

/// List all Types and Statements that needs to be generate for this to work
struct ASTLister : clang::StmtVisitor<ASTLister, void> {
  MLIRGenState &state;
  llvm::DenseSet<clang::QualType> Types;
  std::string TypeStr;
  llvm::raw_string_ostream TypeOS;
  llvm::DenseSet<int /*Stmt::StmtClass*/> Stmts;
  std::string StmtStr;
  llvm::raw_string_ostream StmtOS;

  ASTLister(MLIRGenState &s) : state(s), TypeOS(TypeStr), StmtOS(StmtStr) {}

  void addType(clang::QualType Ty, clang::SourceLocation Loc) {
    if (Ty->isFunctionPointerType())
      return;
    mlir::Type MLIRTy = state.getMLIRType(Ty);
    if (Types.insert(Ty).second) {
      Loc.print(TypeOS, state.ASTctx.getSourceManager());
      TypeOS << "\n";
      MLIRTy.print(TypeOS);
      TypeOS << "\n";
      Ty->dump(TypeOS, state.ASTctx);
      TypeOS << "\n";
    }
  }

  void VitisStmt(clang::DeclRefExpr* DRE) {
    /// This will add the function to the list of functions to "emit". but in
    /// this case functions are not emitted but there statement and types are
    /// listed
    if (auto* FD = dyn_cast<FunctionDecl>(DRE->getDecl()))
      state.GetOrCreateFunc(FD);

    VisitStmt(DRE);
  }

  void VisitStmt(clang::Stmt *S) {
    if (auto *E = dyn_cast<clang::Expr>(S))
      addType(E->getType(), E->getExprLoc());
    if (Stmts.insert(S->getStmtClass()).second) {
      S->dump(StmtOS, state.ASTctx);
      StmtOS << "\n";
    }
    for (auto *Inner : S->children())
      VisitStmt(Inner);
  }

  void print(llvm::raw_ostream &os) {
    os << "Types(" << Types.size() << "):\n";
    os << TypeOS.str();
    os << "\n\nStms(" << Stmts.size() << "):\n";
    os << StmtOS.str();
  }

  void dump() { print(llvm::errs()); }
};

class ArchGenMLIRConsumer : public clang::ASTConsumer {
  clang::CompilerInstance &CI;
  std::unique_ptr<clang::ASTConsumer> llvmirAstConsumer;
  llvm::Module *LLVMModule;
  llvm::LLVMContext *LLVMCtx;
  MLIRGenState state;
  MLIREmitter Visitor;
  ASTLister Lister;

public:
  ArchGenMLIRConsumer(clang::CompilerInstance &CompilerInstance,
                      std::unique_ptr<clang::ASTConsumer> Consumer,
                      llvm::Module *LLVMModule)
      : CI(CompilerInstance), llvmirAstConsumer(std::move(Consumer)),
        LLVMModule(LLVMModule), LLVMCtx(&LLVMModule->getContext()),
        state(CI, LLVMModule), Visitor(state), Lister(state) {}

  void addToEmit(clang::FunctionDecl *FD) {
    if (ListNotEmit)
      Lister.addType(FD->getType(), FD->getLocation());
    else
      state.GetOrCreateFunc(FD);
  }

  void FillFunctionBody(clang::FunctionDecl *FD) {
    if (ListNotEmit)
      Lister.VisitStmt(FD->getBody());
    else
      Visitor.Emit(FD, state.GetOrCreateFunc(FD));
  }

  /// This is transformation fixes function parameters. it is not generic, it
  /// exist only because we do not generate parameter properly.
  /// This is why it is not a pass.
  void rewriteParameters() {
    llvm::SmallVector<mlir::Operation *> maybeDelete;

    state.module->walk([&](approx::ParameterOp op) {
      func::FuncOp func = cast<func::FuncOp>(op->getParentOp());
      auto constantInt =
          cast<arith::ConstantIntOp>(op.input().getDefiningOp());
      int param_idx =
          constantInt.getValue().cast<mlir::IntegerAttr>().getInt() + 1;
      state.builder.setInsertionPointToStart(op->getBlock());
      mlir::Value argVal =
          state.builder
              .create<LLVM::LoadOp>(op.getLoc(),
                                    func.getBody().getArgument(param_idx))
              .getRes();
      op->replaceAllUsesWith(mlir::ValueRange{argVal});

      /// Order matters.
      maybeDelete.push_back(op);
      maybeDelete.push_back(constantInt);
    });
    for (auto *op : maybeDelete)
      if (op->use_empty())
        op->erase();
  }

  mlir::LogicalResult runMLIROptimizationAndLowering() {
    {
      mlir::PassManager pm(&state.ctx);
      pm.addPass(mlir::createInlinerPass());
      if (mlir::failed(pm.run(state.module.get())))
        return mlir::failure();
    }
    rewriteParameters();

    {
      mlir::PassManager pm(&state.ctx);
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      if (mlir::failed(pm.run(state.module.get())))
        return mlir::failure();
    }

    /// This at this point the front-end and cleanup is done
    if (mlir::failed(outputMLIR(MLIROutput, state.module.get())))
      return mlir::failure();

    if (StopAtApprox)
      return mlir::success();

    {
      mlir::PassManager pm(&state.ctx);
      // pm.enableIRPrinting();

      /// The Core of ArchGen, will transform the approx dialect describing the
      /// expression to be approximated into the logic to approximate the
      /// expression
      pm.addPass(approx::createLowerApproxPass());
      /// When debugging it is a lot more useful to see the source locations form
      /// here then from the eval call in the .cpp
      pm.addPass(mlir::createLocationSnapshotPass());

      /// These are not yet correct so they are disabled for now
      // pm.addPass(mlir::createCanonicalizerPass());
      // pm.addPass(mlir::createCSEPass());
      pm.addPass(fixedpt::createConvertFixedPtToArithPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createMemRefToLLVMPass());
      pm.addPass(arith::createConvertArithmeticToLLVMPass());
      pm.addPass(mlir::createConvertFuncToLLVMPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      if (mlir::failed(pm.run(state.module.get())))
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult mergeMLIRintoLLVMIR() {
    std::unique_ptr<llvm::Module> MLIRLLVMModule =
        mlir::translateModuleToLLVMIR(state.module.get(), *LLVMCtx);
    if (!MLIRLLVMModule)
      return mlir::failure();

    MLIRLLVMModule->setDataLayout(LLVMModule->getDataLayout());
    MLIRLLVMModule->setTargetTriple(LLVMModule->getTargetTriple());

    bool brokenDebugInto = false;
    llvm::verifyModule(*MLIRLLVMModule, &llvm::errs(), &brokenDebugInto);

    /// We do not want out flags to collide with the flags from the main llvm
    /// module
    MLIRLLVMModule->getOrInsertModuleFlagsMetadata()->clearOperands();

    llvm::StripDebugInfo(*MLIRLLVMModule);

    for (auto& F : MLIRLLVMModule->functions())
      if (!F.isDeclaration())
        F.addFnAttr(llvm::Attribute::NoInline);

    if (!DontLinkMLIR)
      llvm::Linker::linkModules(*LLVMModule, std::move(MLIRLLVMModule));

    return mlir::success();
  }

  mlir::LogicalResult Finalize() {
    /// List of function we should generate
    std::deque<clang::FunctionDecl *> FuncToEmit = state.takeFuncToEmit();

    /// Go through all the function we need to generate until there is none
    while (!FuncToEmit.empty()) {
      for (auto *FD : FuncToEmit)
        if (FD->getBody())
          FillFunctionBody(FD);
      /// Emitting function bodies may add new functions to generate
      FuncToEmit = state.takeFuncToEmit();
    }

    if (mlir::failed(runMLIROptimizationAndLowering()))
      return mlir::failure();

    if (!StopAtApprox && mlir::failed(mergeMLIRintoLLVMIR()))
      return mlir::failure();

    if (ListNotEmit)
      Lister.dump();
    return mlir::success();
  }

  // Below is the exposed API of clang::ASTConsumer most place just need to
  // forward to LLVMIRGen clang::ASTConsumer
  //===----------------------------------------------------------------------===//
  void Initialize(clang::ASTContext &Context) override {
    return llvmirAstConsumer->Initialize(Context);
  }
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    return llvmirAstConsumer->HandleCXXStaticMemberVarInstantiation(VD);
  }
  bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
    if (llvm::any_of(D, [&](auto *E) {
          return Annot::hasAnnotation(E, Annot::GenAsMLIR);
        })) {
      /// We are assuming here that there the DeclGroupRef conain only code we
      /// should generate or code that the LLVM backend should generate
      assert(llvm::all_of(D, [&](auto *E) {
        return Annot::hasAnnotation(E, Annot::GenAsMLIR);
      }));
      for (auto *E : D)
        addToEmit(cast<clang::FunctionDecl>(E));
      return true;
    } else {
      return llvmirAstConsumer->HandleTopLevelDecl(D);
    }
  }
  void HandleInlineFunctionDefinition(clang::FunctionDecl *D) override {
    return llvmirAstConsumer->HandleInlineFunctionDefinition(D);
  }
  void HandleInterestingDecl(clang::DeclGroupRef D) override {
    return llvmirAstConsumer->HandleInterestingDecl(D);
  }
  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    /// If the frontend emitted an error do not run the backend
    if (!CI.getSema().hasUncompilableErrorOccurred() &&
        mlir::failed(Finalize())) {
      /// If our backend fails stop now
      llvm::report_fatal_error("MLIR pipelined failed");
      return;
    }
    return llvmirAstConsumer->HandleTranslationUnit(Ctx);
  }
  void HandleTagDeclDefinition(clang::TagDecl *D) override {
    return llvmirAstConsumer->HandleTagDeclDefinition(D);
  }
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *D) override {
    return llvmirAstConsumer->HandleTagDeclRequiredDefinition(D);
  }
  void HandleCXXImplicitFunctionInstantiation(clang::FunctionDecl *D) override {
    return llvmirAstConsumer->HandleCXXImplicitFunctionInstantiation(D);
  }
  void HandleTopLevelDeclInObjCContainer(clang::DeclGroupRef D) override {
    return llvmirAstConsumer->HandleTopLevelDeclInObjCContainer(D);
  }
  void HandleImplicitImportDecl(clang::ImportDecl *D) override {
    return llvmirAstConsumer->HandleImplicitImportDecl(D);
  }
  void CompleteTentativeDefinition(clang::VarDecl *D) override {
    return llvmirAstConsumer->CompleteTentativeDefinition(D);
  }
  void CompleteExternalDeclaration(clang::VarDecl *D) override {
    return llvmirAstConsumer->CompleteExternalDeclaration(D);
  }
  void AssignInheritanceModel(clang::CXXRecordDecl *RD) override {
    return llvmirAstConsumer->AssignInheritanceModel(RD);
  }
  void HandleVTable(clang::CXXRecordDecl *RD) override {
    return llvmirAstConsumer->HandleVTable(RD);
  }
  clang::ASTMutationListener *GetASTMutationListener() override {
    return llvmirAstConsumer->GetASTMutationListener();
  }
  clang::ASTDeserializationListener *GetASTDeserializationListener() override {
    return llvmirAstConsumer->GetASTDeserializationListener();
  }
  void PrintStats() override { return llvmirAstConsumer->PrintStats(); }
  bool shouldSkipFunctionBody(clang::Decl *D) override {
    return llvmirAstConsumer->shouldSkipFunctionBody(D);
  }
};

struct NoopASTConsumer : public clang::ASTConsumer {
};

class ArchGenMLIRAction : public clang::PluginASTAction {

  /// Original Main actions being replaced. it usually should be a
  /// clang::CodeGenAction. but not always
  std::unique_ptr<clang::FrontendAction> Inner;
  bool HasLaunchedRealConsumer = false;

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
    /// clang::CodeGenAction should be wrapped but not other kinds of main
    /// actions
    if (auto *CGAct = getInnerAsCodeGenAction()) {
      auto InnerConsumer = Inner->CreateASTConsumer(CI, InFile);
      if (InnerConsumer) {
        HasLaunchedRealConsumer = true;
        return std::make_unique<ArchGenMLIRConsumer>(
            CI, std::move(InnerConsumer),
            CGAct->getCodeGenerator()->GetModule());
      }
    }
    return std::make_unique<NoopASTConsumer>();
  }

  /// Assignments will be set on our clang::FrontendAction and not on the Inner
  /// clang::FrontendAction So before This function copies our data to the Inner
  void copySelfToInner() {
    Inner->setCurrentInput(getCurrentInput(), takeCurrentASTUnit());
    Inner->setCompilerInstance(&getCompilerInstance());
  }

  bool hasIRSupport() const override { return Inner->hasIRSupport(); }

  void ExecuteAction() override {
    if (!HasLaunchedRealConsumer)
      return;
    copySelfToInner();
    return Inner->ExecuteAction();
  }

  void EndSourceFileAction() override {
    if (HasLaunchedRealConsumer)
      return Inner->EndSourceFileAction();
  }

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
