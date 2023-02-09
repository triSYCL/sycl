// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out

#include "sycl/sycl.hpp"

struct Line {
  float slope;
  float intercept;
};

int main() {
  sycl::queue queue{};
  {
    queue.submit([&](sycl::handler &cgh) {
      cgh.single_task([]() { Line l1{0, 0}; });
    });
  }
}
