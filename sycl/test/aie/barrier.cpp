// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int main() {
  aie::device<50, 8> dev;
  aie::queue q(dev);
  q.submit([](auto& ht) {
    ht.single_task([](auto& dt) {
      /// check that we dont dead-lock
      for (int i = 0; i < 100000; i++)
        dt.full_barrier();
    });
  });
}

// CHECK: exit_code=0
