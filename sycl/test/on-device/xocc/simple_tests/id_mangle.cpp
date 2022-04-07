// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*
  Test to see if the world will explode when using SPIR built-ins that are
  derived from those in cl__spirv on Xilinx FPGAs when compiling using xocc.
*/
#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class id_mangle;

// define a host function that will be called on the device and the host to test
// that the correct value is returned for the user defined function and it's not
// replaced by a built-in call and it's not broken in someway.
// Note: This doesn't work on Intel devices at the moment, even using an
// unaltered Intel SYCL compiler, this external get_global_id call seems to
// overwrite the index implementation somehow.
// /todo Look into this? I assumed using SPIRV oriented calls should avoid this
//  interaction. Perhaps a misunderstanding on my part?
#ifdef __SYCL_HAS_XILINX_DEVICE__
size_t get_global_id(uint dimindx) {
  return 1000;
};
#endif

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};

  auto nd = nd_range<3>(range<3>(2, 2, 2), range<3>(1, 1, 1));
  buffer<int, 1> test_buffer{range<1>{8}};

  {
    auto wb = test_buffer.get_access<access::mode::write>();
    for (int i = 0; i < wb.size(); ++i)
      wb[i] = 0;
  }

  q.submit([&](handler &cgh) {
    auto wb = test_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<id_mangle>(nd, [=](nd_item<3> index) {
#ifdef __SYCL_HAS_XILINX_DEVICE__
        wb[index.get_global_linear_id()] += get_global_id(0);
#endif
        wb[index.get_global_linear_id()] += index.get_global_id(0);
        wb[index.get_global_linear_id()] += index.get_global_id(1);
        wb[index.get_global_linear_id()] += index.get_global_id(2);

        wb[index.get_global_linear_id()] += index.get_local_id(0);
        wb[index.get_global_linear_id()] += index.get_local_id(1);
        wb[index.get_global_linear_id()] += index.get_local_id(2);

        wb[index.get_global_linear_id()] += index.get_local_linear_id();

        wb[index.get_global_linear_id()] += index.get_group(0);
        wb[index.get_global_linear_id()] += index.get_group(1);
        wb[index.get_global_linear_id()] += index.get_group(2);

        wb[index.get_global_linear_id()] += index.get_group_linear_id();

        wb[index.get_global_linear_id()] += index.get_group_range(0);
        wb[index.get_global_linear_id()] += index.get_group_range(1);
        wb[index.get_global_linear_id()] += index.get_group_range(2);

        wb[index.get_global_linear_id()] += index.get_global_range(0);
        wb[index.get_global_linear_id()] += index.get_global_range(1);
        wb[index.get_global_linear_id()] += index.get_global_range(2);

        wb[index.get_global_linear_id()] += index.get_local_range(0);
        wb[index.get_global_linear_id()] += index.get_local_range(1);
        wb[index.get_global_linear_id()] += index.get_local_range(2);
    });
  });

  auto rb = test_buffer.get_access<access::mode::read>();

  // The hard coded values tested against here are based on the kernel being
  // executed 8 times (2*2*2) with a local work group size of 1x1x1. Probably
  // possible to make it more flexible with some thought, but this test is
  // mostly about checking for compile errors or run-time ABI problems for
  // missing functions.

  int sum = 0;
  for (int i = 0; i < rb.size(); ++i) {
    sum += rb[i];
  }

  // all of our invocations of the user defined get_global_id on the device plus
  // one host invocation should sum up to 9000, this is only relevant for Xilinx
  // at the moment
#ifdef __SYCL_HAS_XILINX_DEVICE__
  std::cout << "sum of all id's and sizes and user get_global_id call "
            <<  sum + get_global_id(0) << std::endl;
  assert((sum + get_global_id(0)) == 9172);
#else
  std::cout << "sum of all id's, sizes: " <<  sum << std::endl;
  assert(sum == 172);
#endif

  return 0;
}
