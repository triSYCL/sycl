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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class FixedPt_Op<string mnemonic, list<Trait> traits = []> :
    Op<FixedPt_Dialect, mnemonic, traits # [NoSideEffect]> {
  let hasVerifier = 1;
}

class FixedPt_BinOp<string mnemonic, list<Trait> traits = []> : FixedPt_Op<mnemonic, traits> {
  let arguments = (ins
    AnyFixedPt: $lhs,
    AnyFixedPt: $rhs
  );
  let results = (outs AnyFixedPt: $result);
}

def FixedPt_AddOp : FixedPt_BinOp<"add", [Commutative]>;
def FixedPt_SubOp : FixedPt_BinOp<"sub">;
def FixedPt_MulOp : FixedPt_BinOp<"mul", [Commutative]>;
def FixedPt_DivOp : FixedPt_BinOp<"div">;

def FixedPt_ConstantOp : FixedPt_Op<"constant", [ConstantLike]> {
  let arguments = (ins
    FixedPtAttr: $valueAttr
  );
  let results = (outs AnyFixedPt: $result);
  let hasFolder = 1;
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

def FixedPt_RoundOp : FixedPt_Op<"round"> {
  /// TODO: Add rounding semantics: tonearest, tozero, toupper, tolower, toinf...
  let summary = "remove low bits of a fixedpoint type";
  let arguments = (ins
    AnyFixedPt: $input
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$input `:` type($input) `to` type($result) attr-dict";
}

def FixedPt_ExtendOp : FixedPt_Op<"extand"> {
  let summary = "extand high bits and/or low bits of a fixedpoint type (sext or zext)";
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

#endif
