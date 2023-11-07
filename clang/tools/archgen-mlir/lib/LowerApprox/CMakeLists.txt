find_package(FloPoCo COMPONENTS hileco REQUIRED)

# To find functions in sollya by name we need to dlopen it so first we need to find it
find_library(SOLLYA_LIB sollya)

add_mlir_library(ArchGenLowerApprox
  EXCLUDE_FROM_LIBMLIR
  LowerApprox.cpp

  DEPENDS
  ArchGenMLIRApproxPassesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  ArchGenMLIRApprox
  ArchGenMLIRFixedPt
  FloPoCo::hileco
  ${SOLLYA_LIB}
)

target_compile_definitions(ArchGenLowerApprox PRIVATE "ARCHGEN_SOLLYA_LIB_PATH=\"${SOLLYA_LIB}\"")
