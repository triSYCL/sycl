/*
  Test to see if the world will explode when using SPIR built-ins that are
  derived from those in cl__spirv on Xilinx FPGAs when compiling using XOCC.
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
size_t get_global_id(uint dimindx) {
  return 1000;
};

int main() {

  selector_defines::XOCLDeviceSelector xocl;

  queue q{xocl};

  auto nd = nd_range<3>(range<3>(2, 2, 2), range<3>(1, 1, 1));

  buffer<int, 1> test_buffer{range<1>{11}};

  q.submit([&](handler &cgh) {
    auto wb = test_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<id_mangle>(nd, [=](nd_item<3> index) {
        wb[0] += get_global_id(0);

        wb[1] += index.get_global_id(0);
        wb[1] += index.get_global_id(1);
        wb[1] += index.get_global_id(2);

        wb[2] += index.get_global_linear_id();

        wb[3] += index.get_local_id(0);
        wb[3] += index.get_local_id(1);
        wb[3] += index.get_local_id(2);

        wb[4] += index.get_local_linear_id();

        wb[5] += index.get_group(0);
        wb[5] += index.get_group(1);
        wb[5] += index.get_group(2);

        wb[6] += index.get_group_linear_id();

        wb[7] += index.get_group_range(0);
        wb[7] += index.get_group_range(1);
        wb[7] += index.get_group_range(2);

        wb[8] += index.get_global_range(0);
        wb[8] += index.get_global_range(1);
        wb[8] += index.get_global_range(2);

        wb[9] += index.get_local_range(0);
        wb[9] += index.get_local_range(1);
        wb[9] += index.get_local_range(2);

        wb[10] += index.get_offset()[0];
        wb[10] += index.get_offset()[1];
        wb[10] += index.get_offset()[2];
    });
  });

  auto rb = test_buffer.get_access<access::mode::read>();

  // The hard coded values tested against here are based on the kernel being
  // executed 8 times (2*2*2) with a local work group size of 1x1x1. Probably
  // possible to make it more flexible with some thought, but this test is
  // mostly about checking for compile errors or run-time ABI problems for
  // missing functions.

  // all of our invocations of the user defined get_global_id on the device plus
  // one host invocation should sum up to 9000
  printf("get_global_id user defined summation: %d \n", rb[0] + (int)get_global_id(0));
  assert(rb[0] + get_global_id(0) == 9000);

  printf("get_global_id built-in summation: %d \n", rb[1]);
  assert(rb[1] == 12);

  printf("get_global_linear_id built-in summation: %d \n", rb[2]);
  assert(rb[2] == 28);

  printf("get_local_id built-in summation: %d \n", rb[3]);
  assert(rb[3] == 0);

  printf("get_local_linear_id built-in summation: %d \n", rb[4]);
  assert(rb[4] == 0);

  printf("get_group_id built-in summation: %d \n", rb[5]);
  assert(rb[5] == 12);

  printf("get_group_linear_id built-in summation: %d \n", rb[6]);
  assert(rb[6] == 28);

  printf("get_group_range built-in summation: %d \n", rb[7]);
  assert(rb[7] == 48);

  printf("get_global_range built-in summation: %d \n", rb[8]);
  assert(rb[8] == 48);

  printf("get_local_range built-in summation: %d \n", rb[9]);
  assert(rb[9] == 24);

  printf("get_offset built-in summation: %d \n", rb[10]);
  assert(rb[10] == 0);

  return 0;
}
