add_mlir_conversion_library(MLIRFixedPtToArith
  EXCLUDE_FROM_LIBMLIR
  FixedPtToArith.cpp

  DEPENDS
  ArchGenMLIRFixedPtPassesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRArithmeticDialect
  MLIRFuncDialect
  ArchGenMLIRFixedPt
  )
