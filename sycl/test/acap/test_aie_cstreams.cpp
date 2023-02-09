// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin | FileCheck %s -check-prefix CHECK-MERGING
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

// check that we only compile 3 device kernel via chess
// CHECK-MERGING-COUNT-3: Linking Kernel
// CHECK-MERGING-NOT: Linking Kernel

#define TISYCL_DEVICE_ALLOCATOR_DEBUG

#include <sycl/sycl.hpp>

/// Example of using neighbor memory tiles
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace sycl::vendor::xilinx;
using namespace sycl::vendor::xilinx;
using namespace trisycl::vendor::xilinx;

/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
};

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  bool prerun() {
    return true;
  }

  static constexpr int arr_size = 100;
  struct data_type {
    int arr[arr_size];
  };

  void run() {
    if constexpr (t::is_cascade_start()) {
      data_type t;
      for (int i = 0; i < arr_size; i++)
        t.arr[i] = -i;
      t::template cascade_write<data_type>(t);
    } else if constexpr (!t::is_cascade_end()) {
      t::template cascade_write<data_type>(t::template cascade_read<data_type>());
    } else {
      auto v = t::template cascade_read<data_type>();
      for (int i = 0; i < arr_size; i++) {
        assert(v.arr[i] == -i);
      }
    }
  }

  void postrun() {
  }
};

int main(int argc, char **argv) {
  // Define AIE CGRA running a program "prog" on all the tiles of a VC1902
  acap::aie::device<acap::aie::layout::vc1902> aie;
  // acap::aie::device<acap::aie::layout::size<10, 1>> aie;
  // Run up to completion of all the tile programs
  aie.run<prog, tile_memory>();

  return 0;
}
