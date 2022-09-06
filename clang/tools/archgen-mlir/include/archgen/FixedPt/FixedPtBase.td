//===- FixedPtBase.td - Fixed Point attributes and type ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
 
#ifndef ARCHGEN_FIXEDPTBASE_TD
#define ARCHGEN_FIXEDPTBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"
include "mlir/IR/EnumAttr.td"

def FixedPt_Dialect : Dialect {
  let summary = "Dialect to represent fixed point type and operations";
  let name = "fixedpt";
  let cppNamespace = "::archgen::fixedpt";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

// Base class for Func dialect ops.
class FixedPt_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FixedPt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class FixedPt_Attr<string name, string attrMnemonic>
    : AttrDef<FixedPt_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def FixedPtType
    : FixedPt_Type<"FixedPt", "fixedPt", [LLVM_PointerElementTypeInterface]> {

  let parameters = (ins
    "int":$msb,
    "int":$lsb,
    "bool":$is_signed);
  
  let hasCustomAssemblyFormat = true;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isSigned() const { return getIsSigned(); }
    bool isUnsigned() const { return !isSigned(); }
    unsigned getWidth() const { return getMsb() - getLsb() + 1; }
    int getMaxPositiveBW() const { return (isSigned()) ? getMsb() - 1 : getMsb(); }
    llvm::FixedPointSemantics getFixedPointSemantics() const;
    FixedPtType getCommonMulType(FixedPtType other) const;
    FixedPtType getCommonAddType(FixedPtType other) const;
  }];
  let builders = [
    TypeBuilder<(ins "llvm::FixedPointSemantics":$sema), [{
    return Base::get($_ctxt, sema.getWidth() + sema.getLsbWeight() - 1,
                     sema.getLsbWeight(), sema.isSigned());
  }]>];
}

def AnyFixedPt : Type<CPred<"$_self.isa<::archgen::fixedpt::FixedPtType>()">, "fixed point type",
                      "::archgen::fixedpt::FixedPtType">;

def FixedPtAttr : FixedPt_Attr<"FixedPoint", "fixed_point"> {
  let summary = "a fixed point attribute";

  let parameters = (ins "::llvm::APFixedPoint": $value);
  let hasCustomAssemblyFormat = true;
} 

#endif
