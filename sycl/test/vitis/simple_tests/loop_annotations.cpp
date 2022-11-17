// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>

namespace xlx = sycl::ext::xilinx;

int main() {
  sycl::buffer<int> answer{1};
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor a{answer, cgh, sycl::write_only};
    cgh.single_task<class forty_two>([=] {
      for (int i = 0; i < a.get_size(); i++)
        xlx::pipeline([&] { a[i] = 0; });
      for (int i = 0; i < a.get_size(); i++)
        xlx::pipeline<>([&] { a[i] = 0; });
      for (int i = 0; i < a.get_size(); i++)
        xlx::annot<xlx::pipeline<>>([&] { a[i] = 0; });
      for (int i = 0; i < a.get_size(); i++)
        xlx::annot<xlx::pipeline<>, xlx::unroll<>>([&] { a[i] = 0; });
      for (int i = 0; i < a.get_size(); i++)
        xlx::annot<xlx::pipeline<xlx::constrained_ii<1>>,
                   xlx::unroll<xlx::checked_fixed_unrolling<8>>>(
            [&] { a[i] = 0; });
      for (int i = 0; i < a.get_size(); i++)
        xlx::annot<xlx::unroll<xlx::checked_fixed_unrolling<8>>,
                   xlx::dataflow>(
            [&] { a[i] = 0; });
    });
  });
}
