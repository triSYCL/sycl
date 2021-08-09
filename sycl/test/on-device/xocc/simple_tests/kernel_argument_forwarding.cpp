// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out 2>&1 | tee %t.dump
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: cat %t.dump | FileCheck --check-prefix=CHECK1 %s
// RUN: cat %t.dump | FileCheck --check-prefix=CHECK2 %s

#include <CL/sycl.hpp>
#include <CL/sycl/xilinx/fpga.hpp>
#include <array>
#include <iostream>
#include <tuple>
#include <utility>

using namespace sycl::detail::literals;

int main() {
  sycl::buffer<sycl::cl_int, 1> Buffer(4);
  sycl::queue Queue;
  sycl::range<1> NumOfWorkItems{Buffer.get_count()};

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class FirstKernel>(
        sycl::xilinx::kernel_param([=] {
          // CHECK1: v++ {{.*}}class_FirstKernel{{.*}} --optimize 2
          Accessor[0] = 0;
        }, "--optimize 2"_cstr));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class SecondKernel>(sycl::xilinx::kernel_param([=] {
          // CHECK2: v++ {{.*}}class_SecondKernel{{.*}} --kernel_frequency 300
          Accessor[1] = 1;
        }, "--kernel_frequency"_cstr,
        sycl::xilinx::number<0x100 + 0x200, sycl::detail::Base16>::str));
  });
}
