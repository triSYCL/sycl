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
  aie::queue q(dev);
  q.submit_uniform([&](auto& ht) {
    int half_size = buff.size() / 2;
    aie::accessor acc = aie::buffer_range(ht, buff)
                            .read_range(1, 2)
                            .write_range(7, 8);
    ht.single_task([=](auto& dt) {
      /// the read and write range are disjoint so the accessor will also
      /// contain all the data between read and write.
      acc[0] = 4;
      acc[6] = 10;
      assert(acc.size() == 7);
    });
  });
  assert(buff[7] == 10);
}
// CHECK: exit_code=0
