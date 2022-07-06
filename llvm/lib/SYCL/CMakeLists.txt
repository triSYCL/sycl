add_llvm_component_library(LLVMSYCL
  VXXIRDowngrader.cpp
  InSPIRation.cpp
  KernelProperties.cpp
  KernelPropGen.cpp
  PrepareSYCLOpt.cpp
  LowerSYCLMetaData.cpp
  SYCLUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  LINK_COMPONENTS
  Core Support TransformUtils Demangle Analysis
)
