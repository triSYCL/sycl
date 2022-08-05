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

class FixedPt_Op<string mnemonic, list<Trait> traits = []> :
    Op<FixedPt_Dialect, mnemonic, traits>;

class FixedPt_binOp<string mnemonic, list<Trait> traits = []> : FixedPt_Op<mnemonic, traits  # [SameOperandsAndResultType]> {
  let arguments = (ins
    AnyFixedPt: $rhs,
    AnyFixedPt: $lhs
  );
  let results = (outs AnyFixedPt: $result);
  let assemblyFormat = "$rhs `,` $lhs `:` type($result) attr-dict";
}

def FixedPt_addOp : FixedPt_binOp<"add">;
def FixedPt_mulOp : FixedPt_binOp<"mul">;
def FixedPt_subOp : FixedPt_binOp<"sub">;
def FixedPt_divOp : FixedPt_binOp<"div">;
def FixedPt_modOp : FixedPt_binOp<"mod">;

def FixedPt_constantOp : FixedPt_Op<"constant"> {
  let arguments = (ins
    FixedPtAttr: $valueAttr
  );
  let results = (outs AnyFixedPt: $result);
}

def FixedPt_convertOp : FixedPt_Op<"convert"> {
  let summary = "convert a fixedpoint type to an fixedpoint other (trunc, sext, zext)";
  let arguments = (ins
    AnyFixedPt: $input
  );
  let results = (outs AnyFixedPt: $output);
  let assemblyFormat = "$input `:` type($input) `to` type($output) attr-dict";
}

def FixedPt_toIntOp : FixedPt_Op<"toInt"> {
  let summary = "extract the representation of the fixepoint as an integer (bitwidth must match)";
  let arguments = (ins
    AnyFixedPt: $input
  );
  let results = (outs AnyInteger: $output);
  let assemblyFormat = "$input `:` type($input) `as` type($output) attr-dict";
}

def FixedPt_fromIntOp : FixedPt_Op<"fromInt"> {
  let summary = "get a fixepoint from its representation as an integer (bitwidth must match)";
  let arguments = (ins
    AnyInteger: $input
  );
  let results = (outs AnyFixedPt: $output);
  let assemblyFormat = "$input `:` type($input) `as` type($output) attr-dict";
}

#endif
