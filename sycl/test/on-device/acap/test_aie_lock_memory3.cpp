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

graphics::application<float> a;

// All the memory modules are the same
template <typename AIE, int X, int Y>
struct memory : acap::aie::memory<AIE, X, Y> {
  float v;
};

template <typename AIE, int X, int Y>
struct tile : acap::aie::tile<AIE, X, Y> {};

// The tile(0,0) write in memory module to the East
template <typename AIE>
struct tile<AIE, 0, 0> : acap::aie::tile<AIE, 0, 0> {
  using t = acap::aie::tile<AIE, 0, 0>;

  void run() {
    auto &m = t::mem_east();
    auto lock = t::get_lock(acap::hw::dir::east, 0);
    lock.acquire();
    m.v = 0.;
    lock.release_with_value(false);
    while(!a.is_done()) {
      lock.acquire_with_value(false);
      m.v = m.v + 0.1;
      if (m.v > 1.)
        m.v = m.v -2.;
      a.update_tile_data_image(t::x, t::y, &m.v, -1.f, 1.f);
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
      a.update_tile_data_image(t::x, t::y, &m.v, -1.f, 1.f);
      lock.release_with_value(false);
    }
  }
};

int main(int argc, char *argv[]) {
  std::cout << std::endl << "Instantiate small AI Engine:"
            << std::endl << std::endl;
  acap::aie::device<acap::aie::layout::size<2, 1>> aie;

  a.set_device(aie);
  a.start(argc, argv, decltype(aie)::geo::x_size, decltype(aie)::geo::y_size, 1,
          1, 100)
      .image_grid()
      .get_palette()
      .set(graphics::palette::rainbow, 150, 2, 127);

  // Launch the AI Engine program
  aie.run<tile, memory>();
}
