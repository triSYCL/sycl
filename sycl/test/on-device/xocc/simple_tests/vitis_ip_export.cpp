// REQUIRES: xocc

// RUN: %clangxx -std=c++20 --target=vitis_ip-xilinx %s -o %t.zip --vitis-ip-part=xc7vx330t-ffg1157-1 -### 2>&1 | FileCheck %s
// RUN: %clangxx -std=c++20 --target=vitis_ip-xilinx %s -o %t.zip --vitis-ip-part=xc7vx330t-ffg1157-1

// CHECK: clang-{{.*}}"-cc1" "-triple" "vitis_ip-xilinx" "-O3" "-disable-llvm-passes" {{.*}} "-emit-llvm"
// CHECK-NEXT: sycl_vxx.py" "ipexport" "--clang_path" {{.*}} "--target" "xc7vx330t-ffg1157-1"
// CHECK-NOT: clang

#ifndef __VITIS_KERNEL
#error "not using device compiler"
#endif

__VITIS_KERNEL int test(int a, int b) {
  return a + b;
}
