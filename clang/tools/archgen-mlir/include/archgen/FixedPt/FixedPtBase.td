//===- FixedPtBase.td - Fixed Point type and ops -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
 
#ifndef MLIR_DIALECT_FIXEDPTBASE_TD
#define MLIR_DIALECT_FIXEDPTBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def FixedPt_Dialect : Dialect {
  let summary = "Dialect to represent fixed point type and operations";
  let name = "fixedpt";
  let cppNamespace = "::archgen::fixedpt";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class for Func dialect ops.
class FixedPt_Op<string mnemonic, list<Trait> traits = []> :
    Op<FixedPt_Dialect, mnemonic, traits>;

class FixedPt_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FixedPt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class FixedPt_Attr<string name, string attrMnemonic>
    : AttrDef<FixedPt_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def FixedPtType
    : FixedPt_Type<"fixedPt", "fixedPt"> {

  let parameters = (ins
    "int":$msb,
    "int":$lsb,
    "bool":$is_signed);
  
  let assemblyFormat = "`<` $msb `,` $lsb `,` $is_signed `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isSigned() const { return getIsSigned(); }
    bool isUnsigned() const { return !isSigned(); }
    int getWidth() const { return getMsb() - getLsb() + /*width includes both ends*/1; }
    llvm::FixedPointSemantics getFixedPointSemantics() const;
  }];
  let builders = [
    TypeBuilder<(ins "llvm::FixedPointSemantics":$sema), [{
    return Base::get($_ctxt, sema.getWidth() - sema.getScale() /*msb-lsb is an inclusive range*/-1,
                     sema.getScale(), sema.isSigned());
  }]>];
}

def AnyFixedPt : Type<CPred<"$_self.isa<::archgen::fixedpt::fixedPtType>()">, "fixed point",
                      "::archgen::fixedpt::fixedPtType">;

def FixedPtAttr : FixedPt_Attr<"fixedPoint", "fixedPoint"> {
  let summary = "a fixed point attribute";

  let description = [{ 
  }];

  let parameters = (ins "::llvm::APFixedPoint": $value);
  let extraClassDeclaration = [{
  }];

  let hasCustomAssemblyFormat = true;
} 

#endif
