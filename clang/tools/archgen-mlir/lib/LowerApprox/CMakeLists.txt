include_directories(${LLVM_FLOPOCO_INC})

add_mlir_library(ArchGenLowerApprox
  EXCLUDE_FROM_LIBMLIR
  LowerApprox.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRArithmeticDialect
  MLIRFuncDialect
  ArchGenMLIRApprox
  ArchGenMLIRFixedPt
  ${LLVM_FLOPOCO_LIB}
  )
