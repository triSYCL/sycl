// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"
#include <numeric>

int main() {
  constexpr std::size_t size = 10;
  aie::device<1, 1> dev;
  aie::buffer<int> buff(10);
  std::iota(buff.begin(), buff.end(), 0);
  for (int i = 0; i < size; i++)
    assert(buff[i] == i);
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    aie::accessor acc(ht, buff);
    ht.single_task([=](auto& dt) {
      for (int i = 0; i < size; i++) {
        acc[i] *= 2;
      }
    });
  });
  for (int i = 0; i < size; i++)
    assert(buff[i] == i * 2);
}
// CHECK: exit_code=0
