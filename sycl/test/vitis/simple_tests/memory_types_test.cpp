// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-fsycl -std=c++20 -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out > %t.check 2>&1
// RUN: %run_if_not_cpu FileCheck --input-file=%t.check %s
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>
#include <numeric>

int main() {
  sycl::buffer<std::size_t, 1> BufferA{4}, BufferB{4}, BufferC{4};
  sycl::queue Queue{};

  Queue.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::ddr_bank<1>};
    // CHECK-DAG:{{.*}}:DDR[1]
    sycl::accessor Accessor(BufferA, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTesta>([=]{
      std::iota(Accessor.begin(), Accessor.end(), 0);
    });
  });
  Queue.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::hbm_bank<0>};
    // CHECK-DAG:{{.*}}:HBM[0]
    sycl::accessor Accessor(BufferB, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestb>([=]{
      std::iota(Accessor.begin(), Accessor.end(), 0);
    });
  });
  Queue.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::xilinx::plram_bank<1>};
    // CHECK-DAG:{{.*}}:PLRAM[1]
    sycl::accessor Accessor(BufferC, cgh, sycl::write_only, PL);
    cgh.single_task<class SmallerTestc>([=]{
      std::iota(Accessor.begin(), Accessor.end(), 0);
    });
  });
}
