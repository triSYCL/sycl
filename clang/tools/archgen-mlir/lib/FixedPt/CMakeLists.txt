add_mlir_dialect_library(ArchGenMLIRFixedPt
  EXCLUDE_FROM_LIBMLIR
  FixedPt.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  ArchGenMLIRFixedPtIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRDataLayoutInterfaces
  MLIRArithDialect
  )
