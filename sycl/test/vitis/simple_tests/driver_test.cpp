// REQUIRES: xocc

// RUN: %clang -fsycl -fsycl-targets=fpga64_sw_emu -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s
// RUN: env XCL_EMULATION_MODE=sw_emu %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s

// RUN: %clang -fsycl -fsycl-targets=fpga64_hw_emu -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s
// RUN: env XCL_EMULATION_MODE=hw_emu %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s

// RUN: %clang -fsycl -fsycl-targets=fpga64_hw -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s
// RUN: env -u XCL_EMULATION_MODE %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s
// RUN: env XCL_EMULATION_MODE=hw %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE %s

// CHECK-PIPELINE:{{.*}}clang-{{.*}} "-fsycl-is-device"
// CHECK-PIPELINE-NEXT:{{.*}}sycl_vxx.py"
// CHECK-PIPELINE-NEXT:{{.*}}sycl_vxx_post_link.py"
// CHECK-PIPELINE-NEXT:{{.*}}clang-offload-wrapper"
// CHECK-PIPELINE-NEXT:{{.*}}llc"
// CHECK-PIPELINE-NEXT:{{.*}}append-file
// CHECK-PIPELINE-NEXT:{{.*}}clang-{{.*}}" "-cc1" "-triple"{{.*}} "-fsycl-is-host"
// CHECK-PIPELINE-NEXT:{{.*}}ld"
