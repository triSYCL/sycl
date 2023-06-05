// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin

#include "aie.hpp"

int main() {
  struct mem_tile_00 {
    int a00;
  };
  struct mem_tile_01 {
    int a01;
  };
  struct mem_tile_10 {
    int a10;
  };
  struct mem_tile_11 {
    int a11;
  };
  aie::device<2, 2> dev;
  aie::queue q(dev);
  q.submit(
      []<int X, int Y>() {
        if constexpr (X == 0) {
          if constexpr (Y == 0)
            return aie::select<mem_tile_00>();
          else
            return aie::select<mem_tile_01>();
        } else {
          if constexpr (Y == 0)
            return aie::select<mem_tile_10>();
          else
            return aie::select<mem_tile_11>();
        }
      },
      [](auto& ht) {
        ht.single_task([](auto& dt) {
          if constexpr (dt.x() == 0) {
            if constexpr (dt.y() == 0) {
              static_assert(std::is_same_v<decltype(dt.mem()), mem_tile_00&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_north()), mem_tile_01&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_west()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_east()), mem_tile_00&>, "");
              static_assert(dt.pos().get_parity() == aie::hw::parity::east,
                            "");
              dt.mem().a00 = 3;
            } else {
              static_assert(std::is_same_v<decltype(dt.mem()), mem_tile_01&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()), mem_tile_00&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_north()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_west()), mem_tile_01&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_east()), mem_tile_11&>, "");
              static_assert(dt.pos().get_parity() == aie::hw::parity::west,
                            "");
              dt.mem().a01 = 3;
            }
          } else {
            if constexpr (dt.y() == 0) {
              static_assert(std::is_same_v<decltype(dt.mem()), mem_tile_10&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_north()), mem_tile_11&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_west()), mem_tile_00&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_east()), mem_tile_10&>, "");
              static_assert(dt.pos().get_parity() == aie::hw::parity::east,
                            "");
              dt.mem().a10 = 3;
            } else {
              static_assert(std::is_same_v<decltype(dt.mem()), mem_tile_11&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_south()), mem_tile_10&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_north()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(std::is_same_v<decltype(dt.mem_west()), mem_tile_11&>, "");
              static_assert(std::is_same_v<decltype(dt.mem_east()),
                                           aie::detail::out_of_bounds&>,
                            "");
              static_assert(dt.pos().get_parity() == aie::hw::parity::west,
                            "");
              dt.mem().a11 = 3;
            }
          }
        });
      });
}
