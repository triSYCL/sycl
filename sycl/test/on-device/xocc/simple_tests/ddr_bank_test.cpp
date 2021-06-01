// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out 2>&1 | FileCheck %s
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

int main() {
  cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);
  cl::sycl::queue Queue;
  const cl::sycl::cl_int buf_size = Buffer.get_count();

  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::ONEAPI::accessor_property_list PL{sycl::xilinx::ddr_bank<1>};
    // CHECK: :DDR[1]
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTesta>([=]{
          for (cl::sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::ONEAPI::accessor_property_list PL{sycl::xilinx::ddr_bank<3>};
    // CHECK: :DDR[3]
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestb>([=]{
          for (cl::sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::ONEAPI::accessor_property_list PL{sycl::xilinx::ddr_bank<0>};
    // CHECK: :DDR[0]
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestc>([=]{
          for (cl::sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
}
