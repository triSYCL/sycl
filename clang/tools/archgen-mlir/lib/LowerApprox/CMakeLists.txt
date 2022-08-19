find_package(FloPoCo COMPONENTS hileco REQUIRED)

add_mlir_library(ArchGenLowerApprox
  EXCLUDE_FROM_LIBMLIR
  LowerApprox.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  ArchGenMLIRApproxPassesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRArithmeticDialect
  MLIRFuncDialect
  ArchGenMLIRApprox
  ArchGenMLIRFixedPt
  FloPoCo::hileco
)

target_compile_definitions(ArchGenLowerApprox PRIVATE "ARCHGEN_SOLLYA_LIB_PATH=\"${SOLLYA_LIB}\"")
