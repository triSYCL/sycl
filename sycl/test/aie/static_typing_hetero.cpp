// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

template <int X, int Y> struct td_hetero {
  int x;
  int y;
};

int main() {
  aie::device<1, 1> dev;
  aie::queue q(dev);
  q.submit_hetero<td_hetero>([](auto& ht) {
    ht.single_task([](auto& dt) {
      if constexpr (dt.x() == 0) {
        if constexpr (dt.y() == 0) {
          static_assert(std::is_same_v<decltype(dt.mem()), td_hetero<0, 0>&>,
                        "");
        } else {
          static_assert(std::is_same_v<decltype(dt.mem()), td_hetero<0, 1>&>,
                        "");
        }
      } else {
        if constexpr (dt.y() == 0) {
          static_assert(std::is_same_v<decltype(dt.mem()), td_hetero<1, 0>&>,
                        "");
        } else {
          static_assert(std::is_same_v<decltype(dt.mem()), td_hetero<1, 1>&>,
                        "");
        }
      }
    });
  });
}

// CHECK: exit_code=0
