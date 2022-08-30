add_mlir_conversion_library(ArchGenFixedPtToArith
  EXCLUDE_FROM_LIBMLIR
  FixedPtToArith.cpp

  DEPENDS
  ArchGenMLIRFixedPtPassesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRArithmeticDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  ArchGenMLIRFixedPt
  )
