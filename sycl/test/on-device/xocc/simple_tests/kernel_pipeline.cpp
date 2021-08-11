// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -std=c++20 %s -S -emit-llvm -o %t.bundled.ll
// RUN: %clang_offload_bundler --unbundle --type=ll --targets=sycl-%sycl_triple --inputs %t.bundled.ll --outputs %t.bc
// RUN: llvm-dis %t.bc -o=%t.ll
// RUN: cat %t.ll | FileCheck %s

#include <CL/sycl.hpp>
#include <CL/sycl/xilinx/fpga.hpp>
#include <array>
#include <iostream>
#include <tuple>
#include <utility>

using namespace sycl::detail::literals;

int main() {
  constexpr std::size_t len = 120;
  sycl::buffer<sycl::cl_int, 1> Buffer(len);
  sycl::queue Queue;
  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<class FirstKernel>(sycl::xilinx::pipeline_kernel([=] {
      // CHECK-DAG: xilinx_kernel_property
      // CHECK-DAG: kernel_pipeline
      for (size_t i = 0; i < len; ++i)
        Accessor[i] = i;
    }));
  });
}
