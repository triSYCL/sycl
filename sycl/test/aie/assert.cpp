// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int main() {
  aie::device<2, 2> dev;
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    ht.single_task([=](auto& dt) {
      assert(false);
      // clang-format off
// CHECK-DAG: (0, 0): aie(0, 0) at 
// CHECK-DAG: (0, 1): aie(0, 1) at
// CHECK-DAG: (1, 0): aie(1, 0) at
// CHECK-DAG: (1, 1): aie(1, 1) at
      // clang-format on
    });
  });
}
// CHECK: exit_code=134
