// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -std=c++20 %s -o %t.out 2>&1 | tee %t.dump | FileCheck %s

#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>
#include <array>
#include <iostream>
#include <tuple>
#include <utility>

using namespace sycl::ext::xilinx;

template<auto a, typename b>
using numalias = number<a, b>;

int main() {
  sycl::buffer<sycl::cl_int, 1> Buffer(4);
  sycl::queue Queue;
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class FirstKernel>(kernel_param(
        [=] {
          // CHECK-DAG:  {{.*}}v++ {{.*}}class_FirstKernel{{.*}} --optimize 2
          Accessor[0] = 0;
        },
        "--optimize 2"_cstr));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class SecondKernel>(kernel_param(
        [=] {
          // CHECK-DAG: {{.*}}v++ {{.*}}class_SecondKernel{{.*}} --kernel_frequency 300
          Accessor[1] = 1;
        },
        "--kernel_frequency"_cstr ,
        number<0x100 + 0x200, Base16>::str));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class ThirdKernel>("--optimize 2"_vitis_option([=] {
      // CHECK-DAG:  {{.*}}v++ {{.*}}class_ThirdKernel{{.*}} --optimize 2
      Accessor[0] = 0;
    }));
  });
}
