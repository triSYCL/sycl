//===- FuncOps.td - Func operation definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
 
#ifndef MLIR_DIALECT_APPROX_TD
#define MLIR_DIALECT_APPROX_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "archgen/FixedPt/FixedPtBase.td"

def Approx_Dialect : Dialect {
  let summary = "Dialect to represent approximated expression";
  let name = "approx";
  let cppNamespace = "::archgen::approx";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

// Base class for Func dialect ops.
class Approx_Op<string mnemonic, list<Trait> traits = []> :
    Op<Approx_Dialect, mnemonic, traits # [Pure]>;

class Approx_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Approx_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// A TypeAttr for ToBeFolded types.
def ToBeFoldedType
    : Approx_Type<"toBeFolded", "toBeFolded"> {
}

def ApproxAuto: I32EnumAttrCase<"auto_select", 0>;
def ApproxBasicPoly: I32EnumAttrCase<"basic_poly", 1>;
def ApproxBiPartiteTable: I32EnumAttrCase<"bipartite_table", 2>;
def ApproxSimpleTable: I32EnumAttrCase<"simple_table", 3>;

def ApproxKindAttr: I32EnumAttr<"ApproxMode", "describes Approx modes",
                           [ApproxAuto, ApproxBasicPoly, ApproxBiPartiteTable, ApproxSimpleTable]> {
  let cppNamespace = "::archgen::approx";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

def Approx_GenericOp : Approx_Op<"generic"> {
  let arguments = (ins
    Variadic<ToBeFoldedType>: $inputs,
    StrAttr: $action
  );
  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$val, "::llvm::StringRef": $str)>
  ];
  let results = (outs ToBeFoldedType: $output);
  let assemblyFormat = "$action `(` $inputs `)` attr-dict";
}

def Approx_EvaluateOp : Approx_Op<"evaluate"> {
  let arguments = (ins
    ToBeFoldedType: $input,
    ApproxKindAttr: $approx_mode
  );
  let results = (outs FixedPtType: $output);
  let assemblyFormat = "$approx_mode `of` $input `as` type($output) attr-dict";
}

def Approx_VariableOp : Approx_Op<"variable"> {
  let arguments = (ins
    FixedPtType: $input
  );
  let results = (outs ToBeFoldedType: $output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$val)>
  ];
  let assemblyFormat = "$input `:` type($input) attr-dict";
}

def Approx_ParameterOp : Approx_Op<"parameter"> {
  let arguments = (ins
    AnyInteger: $input
  );
  let results = (outs FixedPtType: $output);
  let assemblyFormat = "$input `:` type($input) `as` type($output) attr-dict";
}

def Approx_ConstantOp : Approx_Op<"constant", [ConstantLike]> {
  let arguments = (ins
    FixedPtAttr: $valueAttr
  );
  let results = (outs ToBeFoldedType: $result);
  let extraClassDeclaration = [{
    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::archgen::fixedpt::FixedPointAttr value);
  }];
  let hasFolder = 1;
  let assemblyFormat = "$valueAttr attr-dict";
}

#endif
