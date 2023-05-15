// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin

#include "aie.hpp"

int main() {
  struct A00 {};
  struct A01 {};
  struct A10 {};
  struct A11 {};
  aie::device<2, 2> dev;
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
              static_assert(std::is_same_v<decltype(dt.mem_south()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_north()), A01&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_west()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_east()), A00&>, "");
              static_assert(dt.get_pos().get_parity() == aie::hw::parity::east,
                            "");
            } else {
              static_assert(std::is_same_v<decltype(dt.mem()), A01&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()), A00&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_north()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_west()), A01&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_east()), A11&>, "");
              static_assert(dt.get_pos().get_parity() == aie::hw::parity::west,
                            "");
            }
          } else {
            if constexpr (dt.y() == 0) {
              static_assert(std::is_same_v<decltype(dt.mem()), A10&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_north()), A11&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_west()), A00&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_east()), A10&>, "");
              static_assert(dt.get_pos().get_parity() == aie::hw::parity::east,
                            "");
            } else {
              static_assert(std::is_same_v<decltype(dt.mem()), A11&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()), A10&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_north()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_west()), A11&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_east()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(dt.get_pos().get_parity() == aie::hw::parity::west,
                            "");
            }
          }
        });
      });
}
