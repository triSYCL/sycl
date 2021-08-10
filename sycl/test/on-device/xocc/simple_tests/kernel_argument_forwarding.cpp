// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -std=c++20 %s -o %t.out 2>&1 | tee %t.dump | FileCheck %s

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
    cgh.single_task<class FirstKernel>(sycl::xilinx::kernel_param(
        [=] {
          // CHECK-DAG:  {{.*}}v++ {{.*}}class_FirstKernel{{.*}} --optimize 2
          Accessor[0] = 0;
        },
        "--optimize 2"_cstr));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class SecondKernel>(sycl::xilinx::kernel_param(
        [=] {
          // CHECK-DAG: {{.*}}v++ {{.*}}class_SecondKernel{{.*}} --kernel_frequency 300
          Accessor[1] = 1;
        },
        "--kernel_frequency"_cstr ,
        sycl::xilinx::number<0x100 + 0x200, sycl::detail::Base16>::str));
  });

  Queue.submit([&](sycl::handler &cgh) {
    using namespace sycl::xilinx::literals;
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class ThirdKernel>("--optimize 2"_vitis_option([=] {
      // CHECK-DAG:  {{.*}}v++ {{.*}}class_ThirdKernel{{.*}} --optimize 2
      Accessor[0] = 0;
    }));
  });
}
