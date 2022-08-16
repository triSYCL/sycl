include_directories(${LLVM_FLOPOCO_INC})

# To find functions in sollya by name we need to dlopen it so first we need to find it
find_library(SOLLYA_LIB sollya)

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
  ${LLVM_FLOPOCO_LIB}
  sollya
  mpfr
  gmp
  )

target_compile_definitions(ArchGenLowerApprox PRIVATE "ARCHGEN_SOLLYA_LIB_PATH=\"${SOLLYA_LIB}\"")
