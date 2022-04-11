// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*

  This test is checking if the implementation defined defines created by the
  compiler are being defined correctly for the device and host compilation.

  Some components like the SPIR intrinsics and xilinx includes rely on these
  flags.

  When compiling for a Xilinx device the following defines should be defined:

  __SYCL_HAS_XILINX_DEVICE__ : defined on both the host and device
  __SYCL_SPIR_ONLY__ : defined only on the device

  When compiled for anything but a Xilinx device at the moment both flags should
  be undefined.
*/

#include <CL/sycl.hpp>
#include <iostream>


using namespace cl::sycl;

int main() {
  queue q;

#ifdef __SYCL_SPIR_DEVICE__
        std::cout << "__SYCL_SPIR_DEVICE__ defined on host \n";
        assert(false);
#else
        std::cout << "__SYCL_SPIR_DEVICE__ not defined on host \n";
#endif

#ifdef __SYCL_HAS_XILINX_DEVICE__
        std::cout << "__SYCL_HAS_XILINX_DEVICE__ defined on host \n";
#else
        assert(false);
        std::cout << "__SYCL_HAS_XILINX_DEVICE__ not defined on host \n";
#endif

  buffer<unsigned int> ob(range<1>{2});
  
  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);
    cgh.single_task<class add>([=]() {
#ifdef __SYCL_SPIR_DEVICE__
      wb[0] = 1;
#else
      wb[0] = 2;
#endif

#ifdef __SYCL_HAS_XILINX_DEVICE__
      wb[1] = 1;
#else
      wb[1] = 2;
#endif
    });
  });

   auto rb = ob.get_access<access::mode::read>();
       
   if (rb[0] == 1)   
      std::cout << "__SYCL_SPIR_DEVICE__ defined on device \n";
   else if (rb[0] == 2) 
      std::cout << "__SYCL_SPIR_DEVICE__ not defined on device \n";
   else
      assert("kernel failure");
       
   if (rb[1] == 1)
     std::cout << "__SYCL_HAS_XILINX_DEVICE__ defined on device \n";
   else if (rb[1] == 2)
     std::cout << "__SYCL_HAS_XILINX_DEVICE__ not defined on device \n";
   else
      assert("kernel failure");
  

  return 0;
}
