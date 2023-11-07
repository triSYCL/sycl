// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"
#include <numeric>

int main() {
  constexpr std::size_t size = 10;
  aie::device<2, 3> dev;
  aie::buffer<int> buffs[dev.sizeX][dev.sizeY];
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    auto& b = buffs[ht.x()][ht.y()];
    b.resize(size);
    std::iota(b.begin(), b.end(), 0);
    aie::accessor acc(ht, b);
    ht.single_task([=](auto& dt) {
      for (int i = 0; i < size; i++) {
        acc[i] *= 2;
      }
    });
  });
  for (auto& e : buffs)
    for (auto& b : e)
      for (int i = 0; i < b.size(); i++) {
        std::cout << "buff[" << i << "]=" << b[i] << std::endl;
        assert(b[i] == i * 2);
      }
}
// CHECK: exit_code=0
