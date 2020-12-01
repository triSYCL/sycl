// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out 2>&1 | FileCheck %s
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/xilinx/fpga.hpp>
#include <array>
#include <iostream>
#include <tuple>
#include <utility>

using namespace sycl::xilinx::literal;

int main() {
  sycl::buffer<sycl::cl_int, 1> Buffer(4);
  sycl::queue Queue;
  sycl::range<1> NumOfWorkItems{Buffer.get_count()};

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class FirstKernel>(
        sycl::xilinx::kernel_param("--optimize 2"_cstr, [=] {
    // CHECK: v++ {{.*}}class_FirstKernel{{.*}} --optimize 2
          Accessor[0] = 0;
        }));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class SecondKernel>(sycl::xilinx::kernel_param(
        "--kernel_frequency"_cstr,
        sycl::xilinx::number<0x100 + 0x200, sycl::detail::Base16>::str, [=] {
    // CHECK: v++ {{.*}}class_SecondKernel{{.*}} --kernel_frequency 300
          Accessor[1] = 1;
        }));
  });
}
