#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class id_mangle;


size_t get_global_id(uint dimindx) {
  return 99999999999999;
};

/*
  Test to see if the world will explode when using SPIR built-ins that are
  derived from those in cl__spirv

*/
int main() {

  selector_defines::XOCLDeviceSelector xocl;

  queue q{xocl};

  auto nd = nd_range<3>(range<3>(10, 10, 10), range<3>(2, 2, 2));

  q.submit([&](handler &cgh) {
    cgh.parallel_for<id_mangle>(nd, [=](nd_item<3> index) {

        // problem on device, but not on host. Perhaps just slight rename of
        // host call.
        printf("get_global_id user defined on device: %zu \n", get_global_id(0));
        printf("get_global_id(0): %zu \n", index.get_global_id(0));
        printf("get_global_id(1): %zu \n", index.get_global_id(1));
        printf("get_global_id(2): %zu \n", index.get_global_id(2));
        printf("get_global_linear_id(): %zu \n", index.get_global_linear_id());
        printf("get_local_id(0): %zu \n", index.get_local_id(0));
        printf("get_local_id(1): %zu \n", index.get_local_id(1));
        printf("get_local_id(2): %zu \n", index.get_local_id(2));
        printf("get_local_linear_id(): %zu \n", index.get_local_linear_id());
        printf("get_group(0): %zu \n", index.get_group(0));
        printf("get_group(1): %zu \n", index.get_group(1));
        printf("get_group(2): %zu \n", index.get_group(2));
        printf("get_group_linear_id(): %zu \n", index.get_group_linear_id());
        printf("get_group_range(0): %zu \n", index.get_group_range(0));
        printf("get_group_range(1): %zu \n", index.get_group_range(1));
        printf("get_group_range(2): %zu \n", index.get_group_range(2));
        printf("get_global_range(0): %zu \n", index.get_global_range(0));
        printf("get_global_range(1): %zu \n", index.get_global_range(1));
        printf("get_global_range(2): %zu \n", index.get_global_range(2));
        printf("get_local_range(0): %zu \n", index.get_local_range(0));
        printf("get_local_range(1): %zu \n", index.get_local_range(1));
        printf("get_local_range(2): %zu \n", index.get_local_range(2));
        printf("get_offset()[0]: %zu \n", index.get_offset()[0]);
        printf("get_offset()[1]: %zu \n", index.get_offset()[1]);
        printf("get_offset()[2]: %zu \n", index.get_offset()[2]);
    });
  });

  q.wait();

  printf("get_global_id user defined on host: %zu \n", get_global_id(0));

  return 0;
}
