// REQUIRES: xocc

// RUN: %clang -fsycl -fsycl-targets=fpga64_sw_emu -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_SW_EMU %s
// RUN: env XCL_EMULATION_MODE=sw_emu %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_SW_EMU %s

// RUN: %clang -fsycl -fsycl-targets=fpga64_hw_emu -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_HW_EMU %s
// RUN: env XCL_EMULATION_MODE=hw_emu %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_HW_EMU %s

// RUN: %clang -fsycl -fsycl-targets=fpga64_hw -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_HW %s
// RUN: env -u XCL_EMULATION_MODE %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_HW %s
// RUN: env XCL_EMULATION_MODE=hw %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE -check-prefix=CHECK-PIPELINE_HW %s

// CHECK-PIPELINE_SW_EMU:{{.*}}clang-{{.*}}" "-cc1" "-triple" "fpga64_sw_emu-xilinx-linux"{{.*}}"-fsycl-is-device"{{.*}}"-fsycl-int-header
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}sycl_vxx.py" {{.*}} "sw_emu"
// CHECK-PIPELINE_HW_EMU:{{.*}}clang-{{.*}}" "-cc1" "-triple" "fpga64_hw_emu-xilinx-linux"{{.*}} "-fsycl-is-device"
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}sycl_vxx.py" {{.*}} "hw_emu"
// CHECK-PIPELINE_HW:{{.*}}clang-{{.*}}" "-cc1" "-triple" "fpga64_hw-xilinx-linux"{{.*}} "-fsycl-is-device"
// CHECK-PIPELINE_HW-NEXT:{{.*}}sycl_vxx.py" {{.*}} "hw"
// CHECK-PIPELINE:{{.*}}sycl_vxx_post_link.py"
// CHECK-PIPELINE:{{.*}}clang-offload-wrapper"
// CHECK-PIPELINE:{{.*}}llc"
// CHECK-PIPELINE:{{.*}}append-file
// CHECK-PIPELINE:{{.*}}clang-{{.*}}" "-cc1" "-triple"{{.*}} "-fsycl-is-host"
// CHECK-PIPELINE:{{.*}}ld"
