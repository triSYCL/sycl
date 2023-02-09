// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#include <sycl/sycl.hpp>
#include <cstring>
#include <iostream>
#include <vector>

using namespace sycl::vendor::xilinx;

/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
};

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  /// The lock function of every tiles is run before the prerun of any other
  /// tile.
  void lock() {
  }
  bool prerun() {
#ifndef __SYCL_DEVICE_ONLY__
    xaie::handle h = t::get_dev_handle();
    acap::heap::malloc(h, 0x1044, 45);
    acap::heap::dump_allocator_state(h, 0x1044);
    acap::heap::malloc(h, 0x1044, 78);
    acap::heap::dump_allocator_state(h, 0x1044);
    acap::heap::malloc(h, 0x1044, 768);
    acap::heap::dump_allocator_state(h, 0x1044);
#endif
    return true;
  }

  void run() {
#ifdef __SYCL_DEVICE_ONLY__
    acap::heap::dump_allocator_state();
#endif
  }

  void postrun() {
  }
};

int main(int argc, char **argv) {
  // Define AIE CGRA running a program "prog" on all the tiles of a VC1902
  // acap::aie::device<acap::aie::layout::vc1902> aie;
  acap::aie::device<acap::aie::layout::size<1, 1>> aie;
  // Run up to completion of all the tile programs
  aie.run<prog, tile_memory>();

  return 0;
}
