// REQUIRES: aie
// XFAIL: true

// RUN: %aie_clang %s -o %t.bin
// RUN: %run_on_device %t.bin | FileCheck %s

#include "aie.hpp"

int main() {
  aie::device<2, 2> dev;
  aie::queue q(dev);
  q.submit_uniform([&](auto &ht) {
    ht.single_task([=](auto &dt) mutable {
      assert(false);
// CHECK-DAG: aie(0, 0) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-DAG: aie(1, 0) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-DAG: aie(0, 1) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-DAG: aie(1, 1) at assert.cpp:13: {{.*}} : Assertion `false' failed
// CHECK-NOT: Assertion `false' failed
    });
  });
}
// CHECK: exit_code=0
