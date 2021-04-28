// REQUIRES: xocc

// RUN: %clang -fsycl -fsycl-targets=fpga64_sw_emu -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_SW_EMU %s
// RUN: env XCL_EMULATION_MODE=sw_emu %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_SW_EMU %s
// CHECK-PIPELINE_SW_EMU:{{.*}}clang-12" "-cc1" "-triple" "fpga64_sw_emu-xilinx-linux-sycldevice" "-fsycl" "-fsycl-is-device"
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}sycl-xocc" {{.*}} "sw_emu"
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}clang-offload-wrapper"
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}llc"
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}clang-12" "-cc1" "-triple" "fpga64_sw_emu-xilinx-linux-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-fsycl-int-header
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}clang-12" "-cc1" "-triple" "{{.*}}" {{.*}} "-fsycl" "-fsycl-is-host"
// CHECK-PIPELINE_SW_EMU-NEXT:{{.*}}ld"

// RUN: %clang -fsycl -fsycl-targets=fpga64_hw_emu -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_HW_EMU %s
// RUN: env XCL_EMULATION_MODE=hw_emu %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_HW_EMU %s
// CHECK-PIPELINE_HW_EMU:{{.*}}clang-12" "-cc1" "-triple" "fpga64_hw_emu-xilinx-linux-sycldevice" "-fsycl" "-fsycl-is-device"
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}sycl-xocc" {{.*}} "hw_emu"
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}clang-offload-wrapper"
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}llc"
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}clang-12" "-cc1" "-triple" "fpga64_hw_emu-xilinx-linux-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-fsycl-int-header
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}clang-12" "-cc1" "-triple" "{{.*}}" {{.*}} "-fsycl" "-fsycl-is-host"
// CHECK-PIPELINE_HW_EMU-NEXT:{{.*}}ld"

// RUN: %clang -fsycl -fsycl-targets=fpga64_hw -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_HW %s
// RUN: env -u XCL_EMULATION_MODE %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_HW %s
// RUN: env XCL_EMULATION_MODE=hw %clang -fsycl -fsycl-targets=fpga64 -### %s 2>&1 | FileCheck -check-prefix=CHECK-PIPELINE_HW %s
// CHECK-PIPELINE_HW:{{.*}}clang-12" "-cc1" "-triple" "fpga64_hw-xilinx-linux-sycldevice" "-fsycl" "-fsycl-is-device"
// CHECK-PIPELINE_HW-NEXT:{{.*}}sycl-xocc" {{.*}} "hw"
// CHECK-PIPELINE_HW-NEXT:{{.*}}clang-offload-wrapper"
// CHECK-PIPELINE_HW-NEXT:{{.*}}llc"
// CHECK-PIPELINE_HW-NEXT:{{.*}}clang-12" "-cc1" "-triple" "fpga64_hw-xilinx-linux-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-fsycl-int-header
// CHECK-PIPELINE_HW-NEXT:{{.*}}clang-12" "-cc1" "-triple" "{{.*}}" {{.*}} "-fsycl" "-fsycl-is-host"
// CHECK-PIPELINE_HW-NEXT:{{.*}}ld"
