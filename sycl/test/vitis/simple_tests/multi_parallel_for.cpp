// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*
  The aim of this test is to check that multidimensional kernels are executing
  appropriately.
*/

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>


using namespace cl::sycl;

template <int Dimensions, class kernel_name>
void gen_nd_range(range<Dimensions> k_range, queue q) {
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
    //  std::cout << acc_r[i] << " == " << k_range.size() + i << std::endl;
      assert(acc_r[i] == k_range.size() + i &&
        "incorrect result acc_r[i] != k_range.size() + i");
  }

  q.wait();
}

/*
  Kernel Names:
  par_1d
  par_2d_square
  par_2d_rect
  par_3d_square
  par_3d_rect
*/
// This test does not deal with duplicate kernel names, it was to test the
// ability to extract all the kernels from a file that contained multiple
// kernels in the one translation unit when using xocc (xpirbc consumption path
// doesn't allow you to pass -k all, each kernel needs to be compiled and linked
// separately before being linked together).
int main(int argc, char *argv[]) {
  queue q;

  gen_nd_range<1, class par_1d>({10}, q);
  gen_nd_range<2, class par_2d_square>({10, 10}, q);
  gen_nd_range<2, class par_2d_rect>({12, 6}, q);
  gen_nd_range<3, class par_3d_square>({10, 10, 10}, q);
  gen_nd_range<3, class par_3d_rect>({12, 8, 16}, q);

  return 0;
}
