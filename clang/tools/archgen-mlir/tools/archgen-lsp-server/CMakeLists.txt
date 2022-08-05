
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(ARCHGEN_LSP_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  ArchGenMLIRApprox
  ArchGenMLIRFixedPt
  )

add_llvm_tool(archgen-lsp-server
  archgen-lsp-server.cpp

  DEPENDS
  ${ARCHGEN_LSP_LIBS}
  )
target_link_libraries(archgen-lsp-server PRIVATE ${ARCHGEN_LSP_LIBS})
llvm_update_compile_flags(archgen-lsp-server)
