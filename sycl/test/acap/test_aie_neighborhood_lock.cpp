// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#include <sycl/sycl.hpp>
#include "triSYCL/vendor/Xilinx/acap/aie/hardware.hpp"

using namespace sycl::vendor::xilinx;

/* Programs which propagate an acquire/release pattern in a square
   neighborhood */

// The template define the core tile program by default, so tile(0,0) here
template <typename AIE, int X, int Y> struct tile : acap::aie::tile<AIE, X, Y> {
  void operator()() {
    // Wait for notification from the host
    this->get_lock(0).acquire_with_value(true);
    // Send notification to tile (1,0)
    this->get_lock(acap::hw::dir::east, 1).release_with_value(true);
  }
};

template <typename AIE> struct tile<AIE, 1, 0> : acap::aie::tile<AIE, 1, 0> {
  void operator()() {
    // Wait for notification from tile (0,0)
    this->get_lock(acap::hw::dir::west, 1).acquire_with_value(true);
    // Send notification to tile (1,1)
    this->get_lock(acap::hw::dir::north, 2).release_with_value(true);
  }
};

template <typename AIE> struct tile<AIE, 0, 1> : acap::aie::tile<AIE, 0, 1> {
  void operator()() {
    // Wait for notification from tile (1,1)
    this->get_lock(acap::hw::dir::east, 3).acquire_with_value(true);
    // Send notification to the host
    this->get_lock(4).release_with_value(true);
  }
};

template <typename AIE> struct tile<AIE, 1, 1> : acap::aie::tile<AIE, 1, 1> {
  void operator()() {
    // Wait for notification from tile (1,0)
    this->get_lock(2).acquire_with_value(true);
    // Send notification to tile (0,1)
    this->get_lock(acap::hw::dir::west, 3).release_with_value(true);
  }
};

int main(int argc, char* argv[]) {
  acap::aie::device<acap::aie::layout::size<2, 2>> d;
  // launch the program made of tile
  auto aie_future = d.queue().submit<tile>();
  // Notify the memory module of tile(0,0)
  d.tile(0,0).get_lock(0).release_with_value(true);
  // Alternative API
  d.tile(0,1).get_lock(4).acquire_with_value(true);
  // Wait for all the tiles to complete
  aie_future.get();
}
