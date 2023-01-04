// REQUIRES: vitis

// RUN: %clangxx %EXTRA_COMPILE_FLAGS-fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: %ACC_RUN_PLACEHOLDER %t.out
// TODO should be a Sema test

#include <sycl/sycl.hpp>


using namespace sycl;

class add_2;

int main() {
  queue q;

  q.submit([&] (handler &cgh) {
    cgh.single_task<add_2>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<add_2>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<class add>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<class add>([=] () {
    });
  });

  return 0;
}
