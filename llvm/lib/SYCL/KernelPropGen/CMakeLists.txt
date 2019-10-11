add_llvm_library(LLVMKernelPropGen
  KernelPropGen.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  DEPENDS
  intrinsics_gen
)

target_link_libraries(LLVMKernelPropGen LLVMCore LLVMSupport)
