add_llvm_component_library(LLVMXOCCIRDowngrader
  XOCCIRDowngrader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  DEPENDS
  intrinsics_gen
)
