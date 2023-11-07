// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int main() {
  aie::device<2, 1> dev;
  constexpr std::size_t size = 10;
  aie::buffer<int> buff(size);
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    aie::accessor acc{ht, buff};
    ht.single_task([=](auto& dt) {
      if constexpr (dt.x() == 0) {
        for (int i = 0; i < size; i++) {
          acc[i] = i;
          dt.cascade_write(i);
        }
      } else {
        for (int i = 0; i < size; i++) {
          dt.cascade_read_into(acc[i]);
        }
      }
    });
  });
  for (int i = 0; i < size; i++) {
    assert(buff[i] == i);
  }
}
// CHECK: exit_code=0
