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
include "archgen/FixedPt/FixedPtBase.td"

def Aprox_Dialect : Dialect {
  let summary = "Dialect to represent aproximated expression";
  let name = "aprox";
  let dependentDialects = ["::archgen::fixedpt::FixedPtDialect"];
  let cppNamespace = "::archgen::aprox";
  let useDefaultTypePrinterParser = 1;
}

// Base class for Func dialect ops.
class Aprox_Op<string mnemonic, list<Trait> traits = []> :
    Op<Aprox_Dialect, mnemonic, traits>;

class Aprox_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Aprox_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// A TypeAttr for ToBeFolded types.
def ToBeFoldedType
    : Aprox_Type<"toBeFolded", "toBeFolded"> {
}

def Aprox_genericOp : Aprox_Op<"generic"> {
  let arguments = (ins
    Variadic<AnyType>: $inputs,
    StrAttr: $action
  );
  let results = (outs Variadic<AnyType>: $outputs);
}

def Aprox_constantOp : Aprox_Op<"constant", [ConstantLike]> {
  let arguments = (ins
    FixedPtAttr: $value
  );
  let results = (outs ToBeFoldedType);
  let extraClassDeclaration = [{
    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::archgen::fixedpt::fixedPointAttr value);
  }];
  
}

#endif
