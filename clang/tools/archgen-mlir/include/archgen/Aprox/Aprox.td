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

def Aprox_Dialect : Dialect {
  let summary = "Dialect to represent aproximated expression";

  let name = "aprox";

  let cppNamespace = "::archgen::aprox";
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

#endif
