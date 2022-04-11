// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*
   Regression Test Case/Issue supplied by J-Stephan in issue:
     https://github.com/triSYCL/sycl/issues/69

   It should now function, the problem was the address space being inaccurately
   inferred, fixed by new Infer Pass + additional frontend logic by Intel.

   This test just checks that a simple CMP works correctly, returning the larger
   value to the buffer.
*/

#include <CL/sycl.hpp>

using namespace cl::sycl;

class issue_69;

auto cmp(std::size_t &large, std::size_t small) {
  auto val = (large > small) ? small : large;
  return val;
}

auto main() -> int {
  queue q;

  auto s_buf = cl::sycl::buffer<std::size_t, 3>{cl::sycl::range<3>{5, 5, 5}};
  {
    auto s_w = s_buf.get_access<access::mode::write>();
    for (unsigned int i = 0; i < s_w.get_range()[0]; ++i)
      for (unsigned int j = 0; j < s_w.get_range()[1]; ++j)
        for (unsigned int k = 0; k < s_w.get_range()[2]; ++k)
          s_w[i][j][k] = 42;
  }

  q.submit([&](cl::sycl::handler &cgh) {
    auto s = s_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task<kernel>([=]() {
      // a little bit unusual but trying to keep inline with the original
      // contrived example!
      for (unsigned int i = 0; i < s.get_range()[0]; ++i)
        for (unsigned int j = 0; j < s.get_range()[1]; ++j)
          for (unsigned int k = 0; k < s.get_range()[2]; ++k) {
            const auto id = cl::sycl::id<3>{i, j, k};
            s[id] = cmp(s[id], 40);
          }
    });
  });

  auto s_s = s_buf.get_access<access::mode::read>();
  for (unsigned int i = 0; i < s_s.get_range()[0]; ++i)
    for (unsigned int j = 0; j < s_s.get_range()[1]; ++j)
      for (unsigned int k = 0; k < s_s.get_range()[2]; ++k)
        assert(s_s[i][j][k] == 40 && "invalid result from kernel");

  return 0;
}
