// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-fsycl -std=c++20 -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out > %t.check 2>&1
// RUN: %run_if_not_cpu FileCheck --input-file=%t.check %s
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>

int main() {
  sycl::buffer<sycl::cl_int, 1> BufferA(4), BufferB(4), BufferC(4);
  sycl::queue Queue;
  const sycl::cl_int buf_size = BufferA.size();

  Queue.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::ddr_bank<1>};
    // CHECK-DAG:{{.*}}:DDR[1]
    sycl::accessor Accessor(BufferA, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTesta>([=]{
          for (sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
  Queue.submit([&](sycl::handler &cgh) {
    // CHECK-DAG:{{.*}}:DDR[3]
    sycl::accessor Accessor(BufferB, cgh, sycl::write_only, sycl::ext::oneapi::accessor_property_list{sycl::ext::xilinx::ddr_bank<3>});
    cgh.single_task<class SmallerTestb>([=]{
          for (sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
  Queue.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::ddr_bank<0>};
    // CHECK-DAG:{{.*}}:DDR[0]
    sycl::accessor Accessor(BufferC, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestc>([=]{
          for (sycl::cl_int i = 0 ; i < buf_size ; ++i)
            Accessor[i] = i;
        });
  });
}
