add_llvm_component_library(LLVMSYCL
  XOCCIRDowngrader.cpp
  InSPIRation.cpp
  KernelPropGen.cpp
  PrepareSYCLOpt.cpp
  LowerSYCLMetaData.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  LINK_COMPONENTS
  Passes Core Support TransformUtils Demangle Analysis
)
