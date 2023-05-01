// REQUIRES: aie
// XFAIL: true

// RUN: %aie_clang %s -o %t.bin

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
      [](auto &ht) {
        ht.single_task([](auto &dt) {
          static_assert(decltype(dt)::get_pos().get_parity() == aie::hw::parity::west, "");
          static_assert(decltype(dt)::get_pos() == aie::hw::position{ 0, 0 }, "");
          static_assert(aie::hw::position{ 0, 0 }.get_parity() == decltype(dt)::get_pos().get_parity(), "");
          static_assert(aie::hw::position{ 0, 0 }.get_parity() == aie::hw::parity::east, "");

          // if constexpr (dt.x() == 0) {
          //   if constexpr (dt.y() == 0) {

          //     // static_assert(dt.get_pos().x == 0, "");
          //     // static_assert(dt.get_pos().y == 0, "");
          //     // static_assert(aie::hw::position{ 0, 0 }.moved(aie::hw::dir::west) == aie::hw::position{0, 0}, "");
          //     // static_assert(aie::hw::position{ 0, 0 }.moved(aie::hw::dir::west).x == 0, "");
          //     // static_assert(aie::hw::position{ 0, 0 }.moved(aie::hw::dir::west).y == 0, "");
          //     // static_assert(aie::hw::position{ 0, 0 }.get_parity() == aie::hw::parity::west, "");
          //     // static_assert(aie::hw::get_offset(aie::hw::parity::west, aie::hw::dir::west).x == 0, "");
          //     // static_assert(aie::hw::get_offset(aie::hw::parity::west, aie::hw::dir::west).y == 0, "");
          //     // static_assert(std::is_same_v<decltype(dt.template get_tile_type<aie::hw::dir::west>()), A00 &>, "");

          //     // static_assert(std::is_same_v<decltype(dt.mem()), A00 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_south()), aie::detail::out_of_bounds &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_north()), A01 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_west()), A00 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_east()), A10 &>, "");
          //     // static_assert(dt.get_pos().get_parity() == aie::hw::parity::west, "");
          //   } else {
          //     // static_assert(std::is_same_v<decltype(dt.mem()), A01 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_south()), A00 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_north()), aie::detail::out_of_bounds &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_west()), aie::detail::out_of_bounds &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_east()), A01 &>, "");
          //     // static_assert(dt.get_pos().get_parity() == aie::hw::parity::east, "");
          //   }
          // } else {
          //   if constexpr (dt.y() == 0) {
          //     // static_assert(std::is_same_v<decltype(dt.mem()), A10 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_south()), aie::detail::out_of_bounds &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_north()), A11 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_west()), A00 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_east()), aie::detail::out_of_bounds &>, "");
          //     // static_assert(dt.get_pos().get_parity() == aie::hw::parity::west, "");
          //   } else {
          //     // static_assert(std::is_same_v<decltype(dt.mem()), A11 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_south()), A10 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_north()), aie::detail::out_of_bounds &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_west()), A01 &>, "");
          //     // static_assert(std::is_same_v<decltype(dt.mem_east()), A11 &>, "");
          //     // static_assert(dt.get_pos().get_parity() == aie::hw::parity::east, "");
          //   }
          // }
        });
      });
}
