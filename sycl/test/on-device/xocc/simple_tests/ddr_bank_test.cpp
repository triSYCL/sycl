// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -fsycl -std=c++20 -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out 2>&1 | FileCheck %s
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>

int main() {
  cl::sycl::buffer<cl::sycl::cl_int, 1> BufferA(4), BufferB(4), BufferC(4);
  cl::sycl::queue Queue;
  const cl::sycl::cl_int buf_size = BufferA.size();

  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::ddr_bank<1>};
    // CHECK:{{.*}}:DDR[1]
    sycl::accessor Accessor(BufferA, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTesta>([=]{
          for (cl::sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::ddr_bank<3>};
    // CHECK:{{.*}}:DDR[3]
    sycl::accessor Accessor(BufferB, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestb>([=]{
          for (cl::sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::ddr_bank<0>};
    // CHECK:{{.*}}:DDR[0]
    sycl::accessor Accessor(BufferC, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestc>([=]{
          for (cl::sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
}
