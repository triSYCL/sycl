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

#include "triSYCL/vendor/Xilinx/graphics.hpp"
#include <sycl/sycl.hpp>

using namespace std::chrono_literals;
using namespace sycl::vendor::xilinx;

graphics::application<uint8_t> a;

// All the memory modules are the same
template <typename AIE, int X, int Y>
struct memory : acap::aie::memory<AIE, X, Y> {
  std::uint8_t v;
};

template <typename AIE, int X, int Y> struct tile : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;

  void run() {
    t::log("start\n");
    constexpr bool is_0 = Y == 0;
    std::uint8_t s;
    volatile std::uint8_t *v;
    if constexpr (is_0)
      v = &t::mem_north().v;
    else
      v = &t::mem().v;
#if defined(__SYCL_DEVICE_ONLY__)
    // t::log((int)v);
#endif
    int lock = 15;

    // if constexpr (is_0)
    if constexpr (!is_0)
      *v = 42;
    t::log(*v);
    while (!a.is_done()) {
      t::vertical_barrier();
      if constexpr (is_0)
        *v = (*v + 1) % 255;
      t::log(*v);
      t::vertical_barrier();
      if constexpr (!is_0)
        *v = (*v - 1 + 255) % 255;
      // t::log(*v);
      t::vertical_barrier();
      s = *v;
      t::log(*v);
      a.update_tile_data_image(t::x, t::y, &s, 0, 255);
      // t::vertical_barrier(14);
    }
    t::log("end\n");
  }
  void postrun() {}
};

int main(int argc, char *argv[]) {
  std::cout << std::endl
            << "Instantiate small AI Engine:" << std::endl
            << std::endl;
  acap::aie::device<acap::aie::layout::size<1, 2>> aie;

  a.set_device(aie);
  a.start(argc, argv, decltype(aie)::geo::x_size, decltype(aie)::geo::y_size, 1,
          1, 100)
      .image_grid()
      .get_palette()
      .set(graphics::palette::rainbow, 100, 2, 0);

  // Launch the AI Engine program
  aie.run<tile, memory>();
}
