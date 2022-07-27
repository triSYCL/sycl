get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(ARCHGEN_OPT_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  ArchGenAprox
  ArchGenFixedPt
  )

add_llvm_tool(archgen-opt
  archgen-opt.cpp

  DEPENDS
  ${ARCHGEN_OPT_LIBS}
  )

target_link_libraries(archgen-opt PRIVATE ${ARCHGEN_OPT_LIBS})
llvm_update_compile_flags(archgen-opt)
