/*

  This test is checking if the implementation defined defines created by the
  compiler are being defined correctly for the device and host compilation.

  Some components like the SPIR intrinsics and xilinx includes rely on these
  flags.

  When compiling for a Xilinx device the following defines should be defined:

  __SYCL_XILINX_ONLY__ : defined on both the host and device
  __SYCL_SPIR_ONLY__ : defined only on the device

  When compiled for anything but a Xilinx device at the moment both flags should
  be undefined.
*/

#include <CL/sycl.hpp>
#include <iostream>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};

#ifdef __SYCL_SPIR_DEVICE__
        printf("__SYCL_SPIR_DEVICE__ defined on host \n");
#else
        printf("__SYCL_SPIR_DEVICE__ not defined on host \n");
#endif

#ifdef __SYCL_XILINX_ONLY__
        printf("__SYCL_XILINX_ONLY__ defined on host \n");
#else
        printf("__SYCL_XILINX_ONLY__ not defined on host \n");
#endif

  q.submit([&](handler &cgh) {
    cgh.single_task<class add>([=]() {

#ifdef __SYCL_SPIR_DEVICE__
      printf("__SYCL_SPIR_DEVICE__ defined on device \n");
#else
      printf("__SYCL_SPIR_DEVICE__ not defined on device \n");
#endif

#ifdef __SYCL_XILINX_ONLY__
      printf("__SYCL_XILINX_ONLY__ defined on device \n");
#else
      printf("__SYCL_XILINX_ONLY__ not defined on device \n");
#endif
    });
  });

  q.wait();

  return 0;
}
