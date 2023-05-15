// REQUIRES: aie
// XFAIL: !no_device

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int main() {
  aie::device<2, 2> dev;
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    ht.single_task([=](auto& dt) mutable {
      assert(false);
      // clang-format off
// CHECK-DAG: aie(0, 0) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-DAG: aie(1, 0) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-DAG: aie(0, 1) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-DAG: aie(1, 1) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-NOT: Assertion `false' failed
      // clang-format on
    });
  });
}
// CHECK: exit_code=0
