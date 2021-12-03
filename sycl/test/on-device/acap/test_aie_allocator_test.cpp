// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#include <sycl/sycl.hpp>

#include "triSYCL/vendor/Xilinx/acap/aie/device_allocator.hpp"

#include <cstring>
#include <iostream>
#include <vector>

constexpr unsigned count = 10;

using namespace sycl::vendor::xilinx;
/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
  acap::hw::dev_ptr<void> arr[count];
};

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  /// The lock function of every tiles is run before the prerun of any other
  /// tile.
  bool prerun() {
    return 1;
  }
  void run() {
    auto& arr = t::mem().arr;
    int size_arr[4] = {16, 59, 65, 96};
    int actions[8] = {1, 1, 0, 0, 0, 1, 1, 0};
    int alloc_count = 0;
    int arr_idx = 0;
#ifdef __SYCL_DEVICE_ONLY__
    for (int i = 0; i < (count / 2); i++)
      arr[arr_idx++] = acap::heap::malloc(size_arr[alloc_count++ % 4]);
    for (int i = 0;; i++) {
      if (actions[i % 8])
        arr[arr_idx++] = acap::heap::malloc(size_arr[alloc_count++ % 4]);
      else
        acap::heap::free(arr[arr_idx--].get());
      for (int h = 0; h < arr_idx; h++)
        t::log((uint32_t)arr[h]);
      t::log(i);
      t::log("\n");
    }
#endif
  }

  void postrun() {
#ifndef __SYCL_DEVICE_ONLY__
    acap::hw::dev_ptr<void> arr[count];
    t::get_dev_handle().memcpy_d2h(&arr, acap::hw::offset_table::get_tile_mem_begin_offset(), sizeof(arr));
    int i = 0;
    for (auto e : arr)
      std::cout << std::dec << X << ", " << Y << ", " << i++ << ":" << e.get_int() << std::endl;
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
