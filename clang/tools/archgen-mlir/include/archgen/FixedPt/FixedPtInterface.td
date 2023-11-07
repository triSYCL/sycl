include "mlir/IR/OpBase.td"

def RoundingOpInterface : OpInterface<"RoundingOpInterface"> {
  let description = [{
      Interface to query and update the rounding rules of an operation
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the rounding mode of the operation",
      /*retTy=*/"::archgen::fixedpt::RoundingMode",
      /*methodName=*/"getRoundingMode",
      /*args=*/(ins),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        ConcreteOp op = ::llvm::cast<ConcreteOp>(this->getOperation());
        return op.getRounding();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Set the rounding mode the operation",
      /*retTy=*/"void",
      /*methodName=*/"setRoundingMode",
      /*args=*/(ins "::archgen::fixedpt::RoundingMode": $val),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        ConcreteOp op = ::llvm::cast<ConcreteOp>(this->getOperation());
        op.setRounding(val);
      }]
    >,
  ];
}

