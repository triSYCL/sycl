// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -std=c++20 -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER  %t.dir/exec.out

#include <sycl/sycl.hpp>
#include <iostream>

#include "../utilities/device_selectors.hpp"

int main() {
  // Allocate 1 int of 1D abstract memory
  sycl::buffer<int> answer { 1 };
  // Create a queue on Xilinx FPGA
  sycl::queue q { selector_defines::CompiledForDeviceSelector {} };

  std::cout << "Queue Device: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Queue Device Vendor: "
            << q.get_device().get_info<sycl::info::device::vendor>()
            << std::endl;

  // Submit a kernel on the FPGA
  q.submit([&] (sycl::handler &cgh) {
      // Get a write-only access to the buffer
      sycl::accessor a { answer, cgh, sycl::write_only };
      // The computation on the accelerator
      cgh.single_task([=] { a[0] = 42; });
    });

  // Verify the result
  sycl::host_accessor ans { answer, sycl::read_only };
  std::cout << "The universal answer to the question is " << ans[0]
            << std::endl;
}
