// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin
/// Example of using neighbor memory tiles

#include <sycl/sycl.hpp>

#include "triSYCL/vendor/Xilinx/acap/aie/device_allocator.hpp"

#include <cstring>
#include <iostream>
#include <vector>

static constexpr unsigned count = 6;

using namespace sycl::vendor::xilinx;
/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
  uint32_t arr[count];
};

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  /// The lock function of every tiles is run before the prerun of any other
  /// tile.
  bool prerun() {
    return 1;
  }
  void run() {
#ifdef __SYCL_DEVICE_ONLY__
    __builtin_memset(&t::mem().arr, 0xff, sizeof(t::mem().arr));
    for (int i = 0; i < count; i++)
      t::mem().arr[i] = (uint32_t)acap::heap::malloc(16 + i * 8);
    for (int i = 0; i < count; i++)
      acap::heap::free((void *)t::mem().arr[i]);
#endif
  }

  void postrun() {
#ifndef __SYCL_DEVICE_ONLY__
    uint32_t arr[count];
    t::get_dev_handle().memcpy_d2h(&arr, acap::hw::offset_table::get_tile_mem_begin_offset(), sizeof(arr));
    int i = 0;
    for (uint32_t e : arr)
      std::cout << std::dec << X << ", " << Y << ", " << i++ << ":" << e << std::endl;
#endif
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
