// REQUIRES: vitis && !vitis_cpu

// RUN: %clangxx -std=c++20 --target=vitis_ip-xilinx %s -o %t.zip --vitis-ip-part=xcu200-fsgd2104-2-e -### 2>&1 | FileCheck %s
// RUN: %clangxx -std=c++20 --target=vitis_ip-xilinx %s -o %t.zip --vitis-ip-part=xcu200-fsgd2104-2-e

// CHECK: clang-{{.*}}"-cc1" "-triple" "vitis_ip-xilinx" "-O3" "-disable-llvm-passes" {{.*}}
// CHECK-NEXT: sycl_vxx.py" "ipexport" "--clang_path" {{.*}} "--target" "xcu200-fsgd2104-2-e"
// CHECK-NOT: clang

#ifndef __VITIS_KERNEL
#error "not using device compiler"
#endif

__VITIS_KERNEL int test(int a, int b) {
  return a + b;
}
