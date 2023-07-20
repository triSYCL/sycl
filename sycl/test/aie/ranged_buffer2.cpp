// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"
#include <numeric>

int main() {
  constexpr std::size_t size = 10;
  aie::device<1, 1> dev;
  aie::buffer<int> buff(10, 0);
  std::iota(buff.begin(), buff.begin() + 5, 0);
  for (int i = 0; i < size; i++) {
    std::cout << "buff[" << i << "]=" << buff[i] << std::endl;
  }
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    aie::accessor acc = aie::buffer_range(ht, buff)
                            .read_range(5, buff.size())
                            .write_range(5, buff.size());
    ht.single_task([=](auto& dt) {
      for (int i = 0; i < acc.size(); i++) {
        acc[i] = i + 5;
      }
    });
  });
  for (int i = 0; i < size; i++) {
    std::cout << "buff[" << i << "]=" << buff[i] << std::endl;
    assert(buff[i] == i);
  }
}
// CHECK: exit_code=0
