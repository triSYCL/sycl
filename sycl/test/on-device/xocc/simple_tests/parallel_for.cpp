// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out


/*
  The aim of this test is to check that multidimensional kernels are executing
  appropriately using OpenCL ND range kernels and outputting correct
  information.
*/

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

template <int Dimensions, class kernel_name>
void gen_nd_range(range<Dimensions> k_range) {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};

  buffer<unsigned int> a(k_range.size());

  q.submit([&](handler &cgh) {
    auto acc = a.get_access<access::mode::write>(cgh);

    cgh.parallel_for<kernel_name>(k_range, [=](item<Dimensions> index) {
            unsigned int range = index.get_range()[0];
            for (size_t i = 1; i < Dimensions; ++i)
              range *= index.get_range()[i];

            acc[index.get_linear_id()] = index.get_linear_id() + range;
        });
  });

  auto acc_r = a.get_access<access::mode::read>();

  for (unsigned int i = 0; i < k_range.size(); ++i) {
      // std::cout << acc_r[i] << " == " << k_range.size() + i << std::endl;
      assert(acc_r[i] == k_range.size() + i &&
        "incorrect result acc_r[i] != k_range.size() + i");
  }

  q.wait();
}

int main(int argc, char *argv[]) {
  gen_nd_range<2, class add>({10, 10}); // change the name later..

  return 0;
}
