
set(ARCHGEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Find MLIR directories
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir )
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include )
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)

include_directories(${MLIR_MAIN_INCLUDE_DIR})
include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# If we don't need RTTI or EH, there's no reason to export anything
# from the plugin.
if( NOT MSVC ) # MSVC mangles symbols differently, and
               # ArchGenMLIR.export contains C++ symbols.
  if( NOT LLVM_REQUIRES_RTTI )
    if( NOT LLVM_REQUIRES_EH )
      set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ArchGenMLIR.exports)
    endif()
  endif()
endif()

set(ARCHGEN_PLUGIN_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangCodeGen
  MLIRLLVMDialect
  MLIRArithmeticDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRSupport
  ArchGenMLIRApprox
  ArchGenMLIRFixedPt
  MLIRFixedPtToArith
  MLIRArithmeticToLLVM
  MLIRFuncToLLVM
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRReconcileUnrealizedCasts
  LLVMLinker
)

llvm_add_library(
  ArchGenMLIRPlugin

  ArchGenMLIR.cpp

  MODULE
  DISABLE_LLVM_LINK_LLVM_DYLIB
  
  PLUGIN_TOOL
  clang
  
  LINK_LIBS
  ${ARCHGEN_PLUGIN_LIBS}

  DEPENDS
  ${ARCHGEN_PLUGIN_LIBS}
  )

target_include_directories(ArchGenMLIRPlugin PRIVATE ${MLIR_MAIN_INCLUDE_DIR})
target_include_directories(ArchGenMLIRPlugin PRIVATE ${MLIR_TABLEGEN_OUTPUT_DIR})

add_custom_target(archgen DEPENDS
  archgen-opt
  archgen-lsp-server
  ArchGenMLIRPlugin
  clang
  )

add_subdirectory(test)
