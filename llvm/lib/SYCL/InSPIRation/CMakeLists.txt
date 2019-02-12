add_llvm_library(LLVMInSPIRation
  InSPIRation.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  DEPENDS
  intrinsics_gen
)
