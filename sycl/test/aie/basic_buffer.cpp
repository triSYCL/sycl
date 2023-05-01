// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %run_on_device %t.bin | FileCheck %s

#include "aie.hpp"

int main() {
  constexpr std::size_t size = 10;
  aie::device<3, 3> dev;
  aie::buffer<int> buff;
  for (int i = 0; i < size; i++)
    buff.push_back(i);
  aie::queue q(dev);
  q.submit_uniform([&](auto &ht) {
    aie::accessor acc(ht, buff);
    ht.single_task([=](auto &dt) mutable {
      for (int i = 0; i < size; i++) {
        acc[i] *= 2;
      }
    });
  });
  for (int i = 0; i < size; i++)
    assert(buff[i] == i * 2);
}
// CHECK: exit_code=0
