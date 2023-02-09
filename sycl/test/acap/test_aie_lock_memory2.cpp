// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin
/* Testing the AI Engine Memory Module with locking mechanism

   
*/

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <type_traits>

#include <sycl/sycl.hpp>
#include "triSYCL/vendor/Xilinx/graphics.hpp"

using namespace std::chrono_literals;
using namespace sycl::vendor::xilinx;

graphics::application<uint8_t> a;

// All the memory modules are the same
template <typename AIE, int X, int Y>
struct memory : acap::aie::memory<AIE, X, Y> {
  uint8_t v;
};

// By default all the tiles have an empty program
template <typename AIE, int X, int Y>
struct tile : acap::aie::tile<AIE, X, Y> {};

// The tile(0,0) write in memory module to the East
template <typename AIE>
struct tile<AIE, 0, 0> : acap::aie::tile<AIE, 0, 0> {
  using t = acap::aie::tile<AIE, 0, 0>;

  void run() {
    auto &m = t::mem_east();
    auto lock = t::get_lock(acap::hw::dir::east, 0);
    m.v = 42;
    while (!a.is_done()) {
      lock.acquire_with_value(false);
      ++m.v;
      a.update_tile_data_image(t::x, t::y, &m.v, 42, 143);
      lock.release_with_value(true);
    }
  }
};

// The tile(1,0) read from memory module to the West
template <typename AIE>
struct tile<AIE, 1, 0> : acap::aie::tile<AIE, 1, 0> {
  using t = acap::aie::tile<AIE, 1, 0>;

  void run() {
    auto &m = t::mem_west();
    auto lock = t::get_lock(acap::hw::dir::west, 0);
    while (!a.is_done()) {
      lock.acquire_with_value(true);
      a.update_tile_data_image(t::x, t::y, &m.v, 42, 143);
      lock.release_with_value(false);
    }
  }
};

int main(int argc, char *argv[]) {
  std::cout << std::endl << "Instantiate small AI Engine:"
            << std::endl << std::endl;
  acap::aie::device<acap::aie::layout::size<2, 1>> aie;

  a.set_device(aie);
  a.start(argc, argv, decltype(aie)::geo::x_size,
          decltype(aie)::geo::y_size, 1, 1, 100);

  // Launch the AI Engine program
  aie.run<tile, memory>();
}
