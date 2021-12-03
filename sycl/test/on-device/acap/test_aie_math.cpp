// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin
/// Example of using neighbor memory tiles

#include <iomanip>
#include <sycl/sycl.hpp>

#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>

using in_type = float;
using out_type = float;

static constexpr int count = 10;

using namespace sycl::vendor::xilinx;
/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
  volatile out_type out[count];
  volatile in_type in[count];
};

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  bool prerun() {
    out_type out[count];
    in_type in[count];
    in_type d = 3.0;
    for (int i = 0; i < count; i++) {
      in[i] = d;
      d *= 1.5;
    }
    t::get_dev_handle().memcpy_h2d(
        acap::hw::offset_table::get_tile_mem_begin_offset() + sizeof(out),
        (in_type *)&in, sizeof(in));
    return 1;
  }

  out_type out[count];
  in_type in[count];

  void run() {
#ifdef __SYCL_DEVICE_ONLY__
    auto& in = t::mem().in;
    auto& out = t::mem().out;
#endif
    for (int i = 0; i < count; i++) {
      out[i] = in[i] - in[(i + 1) % count];
    }
  }

  void postrun() {
    out_type dev_out[count];
    t::get_dev_handle().memcpy_d2h(
        &dev_out, acap::hw::offset_table::get_tile_mem_begin_offset(),
        sizeof(dev_out));
    run();
    out_type avg_delta = 0;
    for (int i = 0; i < count; i++) {
      bool is_ok = false;
      if constexpr (std::is_floating_point_v<out_type>) {
        constexpr out_type epsilon = 0.01;
        avg_delta = (avg_delta * i + std::abs(dev_out[i] - out[i])) / (i + 1);
        is_ok = (std::abs(dev_out[i] - out[i]) < epsilon);
      } else
        is_ok = (dev_out[i] == out[i]);
    }
    std::cout << "avg_delta = " << std::fixed << std::showpoint
              << std::setprecision(10) << avg_delta << "\n";
    if (avg_delta > 0.1) {
      std::cout << "TEST FAILED" << std::endl;
      assert(false);
    }
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
