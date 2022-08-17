//===- FuncOps.td - Func operation definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
 
#ifndef MLIR_DIALECT_APROX_TD
#define MLIR_DIALECT_APROX_TD

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
    Op<Approx_Dialect, mnemonic, traits # [NoSideEffect]>;

class Approx_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Approx_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// A TypeAttr for ToBeFolded types.
def ToBeFoldedType
    : Approx_Type<"toBeFolded", "toBeFolded"> {
}

def Approx_GenericOp : Approx_Op<"generic"> {
  let arguments = (ins
    Variadic<AnyType>: $inputs,
    StrAttr: $action
  );
  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$val, "::llvm::StringRef": $str)>
  ];
  let results = (outs ToBeFoldedType: $output);
}

def Approx_EvaluateOp : Approx_Op<"evaluate"> {
  let arguments = (ins
    ToBeFoldedType: $input
  );
  let results = (outs FixedPtType: $output);
}

def Approx_VariableOp : Approx_Op<"variable"> {
  let arguments = (ins
    FixedPtType: $input
  );
  let results = (outs ToBeFoldedType: $output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$val)>
  ];
}

def Approx_ParameterOp : Approx_Op<"parameter"> {
  let arguments = (ins
    AnyInteger: $input
  );
  let results = (outs FixedPtType: $output);
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
}

#endif
