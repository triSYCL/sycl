// REQUIRES: xocc

// RUN: %clang -fsycl -fsycl-targets=fpga64_hls_sw_emu -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-SW_EMU %s
// env XCL_EMULATION_MODE=hls_sw_emu %clang -fsycl -fsycl-targets=fpga64 -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-SW_EMU %s

// RUN: %clang -fsycl -fsycl-targets=fpga64_hls_hw_emu -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-HW_EMU %s
// env XCL_EMULATION_MODE=hls_hw_emu %clang -fsycl -fsycl-targets=fpga64 -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-HW_EMU %s

// RUN: %clang -fsycl -fsycl-targets=fpga64_hls_hw -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-HW %s
// env -u XCL_EMULATION_MODE %clang -fsycl -fsycl-targets=fpga64 -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-HW %s
// env XCL_EMULATION_MODE=hls_hw %clang -fsycl -fsycl-targets=fpga64 -dM -E %s 2>&1 | FileCheck -check-prefix=CHECK-MACRO-COMMON -check-prefix=CHECK-MACRO-HW %s

// CHECK-MACRO-COMMON-DAG: #define __SYCL_HAS_XILINX_DEVICE__ 1
// CHECK-MACRO-COMMON-DAG: #define __XilinxHLS 1
// CHECK-MACRO-COMMON-DAG: #define __XilinxHLS64 1
// CHECK-MACRO-COMMON-DAG: #define __XilinxHLS64__ 1
// CHECK-MACRO-COMMON-DAG: #define __XilinxHLS__ 1
// CHECK-MACRO-SW_EMU-DAG: #define __SYCL_XILINX_SW_EMU_MODE__ 1
// CHECK-MACRO-HW_EMU-DAG: #define __SYCL_XILINX_HW_EMU_MODE__ 1
// CHECK-MACRO-HW-DAG: #define __SYCL_XILINX_HW_MODE__ 1
