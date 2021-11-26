add_llvm_component_library(LLVMSYCL
  ChessMassage.cpp
  InSPIRation.cpp
  KernelProperties.cpp
  KernelPropGen.cpp
  PrepareSYCLOpt.cpp
  PrepareSyclChessOpt.cpp
  LowerSYCLMetaData.cpp
  VXXIRDowngrader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCL

  LINK_COMPONENTS
  Passes Core Support TransformUtils Demangle Analysis
)
