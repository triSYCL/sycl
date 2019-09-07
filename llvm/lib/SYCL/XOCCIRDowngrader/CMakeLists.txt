add_llvm_library(LLVMXOCCIRDowngrader
  XOCCIRDowngrader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  DEPENDS
  intrinsics_gen
)

target_link_libraries(LLVMXOCCIRDowngrader LLVMCore LLVMSupport)
