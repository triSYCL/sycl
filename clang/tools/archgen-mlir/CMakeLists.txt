
# Find MLIR directories
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include )
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include )
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)

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

set(LINK_LIBRARIES
  clangAST
  clangBasic
  clangFrontend
  clangCodeGen
  MLIRIR
)

llvm_add_library(
  ArchGenMLIR

  ArchGenMLIR.cpp

  MODULE
  DISABLE_LLVM_LINK_LLVM_DYLIB
  
  PLUGIN_TOOL
  clang
  
  LINK_LIBS
  ${LINK_LIBRARIES}

  DEPENDS
  ${LINK_LIBRARIES}
  )

target_include_directories(ArchGenMLIR PRIVATE ${MLIR_INCLUDE_DIR})
target_include_directories(ArchGenMLIR PRIVATE ${MLIR_TABLEGEN_OUTPUT_DIR})
