// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  sycl::buffer<int> answer{1};
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor a{answer, cgh, sycl::write_only};
    cgh.single_task([=] { a[0] = 42; });
  });

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor a{answer, cgh, sycl::write_only};
    cgh.single_task<class test>([=] { a[1] = 42; });
  });

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor a{answer, cgh, sycl::write_only};
    cgh.parallel_for(sycl::range<1>(1),
                     [=](sycl::id<1> item) { a[item] = 42; });
  });

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor a{answer, cgh, sycl::write_only};
    cgh.parallel_for<class test2>(sycl::range<1>(1),
                                  [=](sycl::id<1> item) { a[item] = 42; });
  });
}
