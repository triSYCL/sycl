// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#define TRISYCL_DEVICE_ALLOCATOR_DEBUG
#define TRISYCL_DEVICE_STREAM_DEBUG

/// Example of using neighbor memory tiles
#include <sycl/sycl.hpp>

#include "triSYCL/vendor/Xilinx/graphics.hpp"
#include "triSYCL/vendor/Xilinx/acap/aie/rpc.hpp"
#include "triSYCL/vendor/Xilinx/acap/aie/log.hpp"

#include "triSYCL/vendor/Xilinx/acap/aie/device_libstdcpp.hpp"

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
  /// The lock function of every tiles is run before the prerun of any other
  /// tile.
  bool prerun() {
    return 1;
  }

  static constexpr int arr_size = 10;
  struct data_type {
    int arr[arr_size];
  };

  void run() {
    data_type t;
    for (int i = 0; i < arr_size; i++)
      t.arr[i] = i;
    t::template stream_write<data_type>(t, 0);
    data_type v = t::template stream_read<data_type>(0);
    for (int i = 0; i < arr_size; i++) {
      assert(v.arr[i] == i);
    }
  }

  void postrun() {
  }
};

int main(int argc, char **argv) {
  // Define AIE CGRA running a program "prog" on all the tiles of a VC1902
  // acap::aie::device<acap::aie::layout::vc1902> aie;
  acap::aie::device<acap::aie::layout::size<1, 1>> aie;

  aie.tile(0, 0).connect(decltype(aie)::csp::me_0, decltype(aie)::cmp::north_0);
  aie.tile(0, 1).connect(decltype(aie)::csp::south_0, decltype(aie)::cmp::south_1);
  aie.tile(0, 0).connect(decltype(aie)::csp::north_1, decltype(aie)::cmp::me_0);

  // Run up to completion of all the tile programs
  aie.run<prog, tile_memory>();

  return 0;
}
