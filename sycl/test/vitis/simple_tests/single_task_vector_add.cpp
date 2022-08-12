// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out
// Test that we can get a fast reports by adding --vitis-ip-part
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec2.out --vitis-ip-part=xcu200-fsgd2104-2-e

/*
   A simple typical FPGA-like kernel adding 2 vectors
*/
#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>


using namespace sycl;

constexpr size_t N = 300;
using Type = int;

int main(int argc, char *argv[]) {
  buffer<Type> a { N };
  buffer<Type> b { N };
  buffer<Type> c { N };

  {
    auto a_b = b.get_access<access::mode::discard_write>();
    // Initialize buffer with increasing numbers starting at 0
    std::iota(&a_b[0], &a_b[a_b.size()], 0);
  }

  {
    auto a_c = c.get_access<access::mode::discard_write>();
    // Initialize buffer with increasing numbers starting at 5
    std::iota(&a_c[0], &a_c[a_c.size()], 5);
  }

  queue q;

  std::cout << "Queue Device: " << q.get_device().get_info<info::device::name>() << std::endl;
  std::cout << "Queue Device Vendor: " << q.get_device().get_info<info::device::vendor>() << std::endl;

  // Launch a kernel to do the summation
  q.submit([&] (handler &cgh) {
      // Get access to the data
      auto a_a = a.get_access<access::mode::write>(cgh);
      auto a_b = b.get_access<access::mode::read>(cgh);
      auto a_c = c.get_access<access::mode::read>(cgh);

      // A typical FPGA-style pipelined kernel
      cgh.single_task<class add>([=] () {
          // Use an intermediate automatic array
          decltype(a_b)::value_type array[N];
          for (unsigned int i = 0 ; i < N; ++i)
            array[i] = a_b[i];
          for (unsigned int i = 0 ; i < N; ++i)
            a_a[i] = array[i] + a_c[i];
        });
    });

  // Verify the result
  auto a_a = a.get_access<access::mode::read>();
  for (unsigned int i = 0 ; i < a.size(); ++i) {
    assert(a_a[i] == 5 + 2*i && "invalid result from kernel");
  }

  return 0;
}
