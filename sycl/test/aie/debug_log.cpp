// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %run_on_device %t.bin | FileCheck %s

#include "aie.hpp"

int main() {
  aie::device<1, 1> dev;
  aie::queue q(dev);
  q.submit([](auto &ht) {
    ht.single_task([](auto &dt) {
#ifdef __SYCL_DEVICE_ONLY__
      aie::detail::debug_log("abcdefghijklmnopqrstuvwxyz\n");
#endif
// CHECK: abcdefghijklmnopqrstuvwxyz
    });
  });
}
// CHECK: exit_code=0
