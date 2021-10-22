// The universal answer on an heterogeneous world

#include <sycl/sycl.hpp>
#include <iostream>

#include "../utilities/device_selectors.hpp"

int main() {
  // Allocate some abstract memory
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
      auto a = answer.get_access<sycl::access::mode::write>(cgh);
      // The computation on the accelerator
      cgh.single_task<class forty_two>([=] { a[0] = 42; });
    });

  // Verify the result
  auto ans = answer.get_access<sycl::access::mode::read>();
  std::cout << "The universal answer to the question is " << ans[0]
            << std::endl;
}
