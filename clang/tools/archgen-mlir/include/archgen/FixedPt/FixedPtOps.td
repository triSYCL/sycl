//===- FixedPt.td - Fixed Point ops ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARCHGEN_FIXEDPT_TD
#define ARCHGEN_FIXEDPT_TD

include "archgen/FixedPt/FixedPtBase.td"
include "archgen/FixedPt/FixedPtInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def RoundDown: I32EnumAttrCase<"truncate", 0>;
def RoundNearest: I32EnumAttrCase<"nearest", 1>;
def RoundNearestEvenToUp: I32EnumAttrCase<"nearest_even_to_up", 2>;
def RoundNearestEvenToDown: I32EnumAttrCase<"nearest_even_to_down", 3>;

def RoundingKindAttr: I32EnumAttr<"RoundingMode", "describes rounding modes",
                           [RoundDown, RoundNearest, RoundNearestEvenToUp, RoundNearestEvenToDown]> {
  let cppNamespace = "::archgen::fixedpt";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

class FixedPt_Op<string mnemonic, list<Trait> traits = []> :
    Op<FixedPt_Dialect, mnemonic, traits # [NoSideEffect]> {
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

class FixedPt_BinOp<string mnemonic, list<Trait> traits = []> : 
FixedPt_Op<mnemonic, traits # [DeclareOpInterfaceMethods<RoundingOpInterface>]> {
  let arguments = (ins
    AnyFixedPt: $lhs,
    AnyFixedPt: $rhs,
    RoundingKindAttr: $rounding
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) $rounding type($result) attr-dict";
}

class FixedPt_VarOp<string mnemonic, list<Trait> traits = []> :
    FixedPt_Op<mnemonic, traits # [DeclareOpInterfaceMethods<RoundingOpInterface>]> {
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let arguments = (ins
    RoundingKindAttr: $rounding,
    Variadic<AnyFixedPt>: $args
  );
  let results = (outs AnyFixedPt: $result);
  let extraClassDeclaration = [{
    ::mlir::Value lhs();
    ::mlir::Value rhs();
    ::mlir::MutableOperandRange lhsMutable();
    ::mlir::MutableOperandRange rhsMutable();
  }];
  let extraClassDefinition  = [{
::mlir::Value $cppClass::lhs() {
  assert(getNumOperands() == 2);
  return *getODSOperands(0).begin();
}

::mlir::Value $cppClass::rhs() {
  assert(getNumOperands() == 2);
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange $cppClass::lhsMutable() {
  assert(getNumOperands() == 2);
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange $cppClass::rhsMutable() {
  assert(getNumOperands() == 2);
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}
  }];
  let hasCustomAssemblyFormat = true;
}

def FixedPt_AddOp : FixedPt_VarOp<"add", [Commutative]>;
def FixedPt_SubOp : FixedPt_BinOp<"sub">;
def FixedPt_MulOp : FixedPt_VarOp<"mul", [Commutative]>;
def FixedPt_DivOp : FixedPt_BinOp<"div">;

def FixedPt_ConstantOp : FixedPt_Op<"constant", [ConstantLike]> {
  let arguments = (ins
    FixedPtAttr: $valueAttr
  );
  let results = (outs AnyFixedPt: $result);
  let hasFolder = 1;
  let hasCustomAssemblyFormat = true;
}

def FixedPt_TruncOp : FixedPt_Op<"trunc"> {
  /// TODO: Add truncating semantics: saturate or modular
  let summary = "truncate the high bits of a fixedpoint type (may change sign)";
  let arguments = (ins
    AnyFixedPt: $input
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$input `:` type($input) `to` type($result) attr-dict";
}

def FixedPt_RoundOp : FixedPt_Op<"round", [DeclareOpInterfaceMethods<RoundingOpInterface>]> {
  /// TODO: Add rounding semantics: tonearest, tozero, toupper, tolower, toinf...
  let summary = "remove low bits of a fixedpoint type";
  let arguments = (ins
    AnyFixedPt: $input,
    RoundingKindAttr: $rounding
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$input `:` type($input) $rounding type($result) attr-dict";
}

def FixedPt_ExtendOp : FixedPt_Op<"extend"> {
  let summary = "extend high bits and/or low bits of a fixedpoint type (sext or zext)";
  let arguments = (ins
    AnyFixedPt: $input
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$input `:` type($input) `to` type($result) attr-dict";
}

def FixedPt_BitcastOp : FixedPt_Op<"bitcast"> {
  let summary = "reinterpret the representation as an other type (width must match)";
  let arguments = (ins
    AnyType: $input
  );
  let results = (outs AnyType: $result);
  let assemblyFormat = "$input `:` type($input) `as` type($result) attr-dict";
}

def FixedPt_ConvertOp : FixedPt_Op<"convert", [DeclareOpInterfaceMethods<RoundingOpInterface>]> {
  let summary = "convert by any means necessary";
  let arguments = (ins
    AnyFixedPt: $input,
    RoundingKindAttr: $rounding
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$input `:` type($input) $rounding type($result) attr-dict";
}

#endif
