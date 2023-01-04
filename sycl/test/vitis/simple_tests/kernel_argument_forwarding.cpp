// REQUIRES: vitis

// RUN: %clangxx %EXTRA_COMPILE_FLAGS-fsycl -fsycl-targets=%sycl_triple -std=c++20 %s -o %t.out > %t.check 2>&1
// RUN: %run_if_not_cpu FileCheck --input-file=%t.check %s

#include <sycl.hpp>
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
          // CHECK-DAG:  {{.*}}v++ {{.*}}FirstKernel{{.*}} --kernel_frequency 200
          Accessor[0] = 0;
        },
        "--kernel_frequency 200"_cstr));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class SecondKernel>(kernel_param(
        [=] {
          // CHECK-DAG: {{.*}}v++ {{.*}}SecondKernel{{.*}} --kernel_frequency 300
          Accessor[1] = 1;
        },
        "--kernel_frequency"_cstr ,
        number<0x100 + 0x200, Base16>::str));
  });

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class ThirdKernel>("--kernel_frequency 300"_vitis_option([=] {
      // CHECK-DAG:  {{.*}}v++ {{.*}}ThirdKernel{{.*}} --kernel_frequency 300
      Accessor[0] = 0;
    }));
  });
}
