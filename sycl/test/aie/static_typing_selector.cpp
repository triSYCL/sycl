// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int main() {
  struct A00 {};
  struct A01 {};
  struct A10 {};
  struct A11 {};
  aie::device<1, 1> dev;
  aie::queue q(dev);
  q.submit(
      []<int X, int Y>() {
        if constexpr (X == 0) {
          if constexpr (Y == 0)
            return aie::select<A00>();
          else
            return aie::select<A01>();
        } else {
          if constexpr (Y == 0)
            return aie::select<A10>();
          else
            return aie::select<A11>();
        }
      },
      [](auto& ht) {
        ht.single_task([](auto& dt) {
          if constexpr (dt.x() == 0) {
            if constexpr (dt.y() == 0) {
              static_assert(std::is_same_v<decltype(dt.mem()), A00&>, "");
            } else {
              static_assert(std::is_same_v<decltype(dt.mem()), A01&>, "");
            }
          } else {
            if constexpr (dt.y() == 0) {
              static_assert(std::is_same_v<decltype(dt.mem()), A10&>, "");
            } else {
              static_assert(std::is_same_v<decltype(dt.mem()), A11&>, "");
            }
          }
        });
      });
}

// CHECK: exit_code=0
