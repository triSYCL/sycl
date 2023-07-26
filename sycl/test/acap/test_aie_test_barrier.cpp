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
#include <random>

using namespace sycl::vendor::xilinx;

constexpr int count = 4;

using in_type = float;
using out_type = float;

/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
};

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  // bool prerun() {
  //   in_type d = 3.0;
  //   for (int i = 0; i < count; i++) {
  //     in0[i] = global_in0[i];
  //     in1[i] = global_in1[i];
  //     // in[i] = d;
  //     // d *= 1.5;
  //   }
  //   num = count;
  //   t::get_dev_handle().memcpy_h2d(acap::hw::args_begin_offset + sizeof(out),
  //                                  (in_type *)&in0,
  //                                  sizeof(in0) + sizeof(in1) + sizeof(num));
  //   return 1;
  // }
  volatile out_type out[count];
  volatile in_type in0[count];
  volatile in_type in1[count];
  int num;
  void run() {
    for (int i = 0; i < 10; i++)
      t::barrier();
  }

  // void postrun() {
  //   out_type dev_out[count];
  //   constexpr out_type epsilon = 0.001;
  //   t::get_dev_handle().memcpy_d2h(&dev_out, acap::hw::args_begin_offset,
  //                                  sizeof(dev_out));
  //   run();
  //   out_type avg_delta = 0;
  //   for (int i = 0; i < count; i++) {
  //     bool is_ok = false;
  //     if constexpr (std::is_floating_point_v<out_type>) {
  //       avg_delta = (avg_delta * i + std::abs(dev_out[i] - out[i])) / (i + 1);
  //       is_ok = (std::abs(dev_out[i] - out[i]) < epsilon);
  //     } else
  //       is_ok = (dev_out[i] == out[i]);
  //   }
  //   std::cout << "avg_delta = " << std::fixed << std::showpoint
  //             << std::setprecision(10) << avg_delta << "\n";
  //   if (avg_delta > epsilon) {
  //     std::cout << "root_seed = " << seed0 << " iteration = " << iter << " tmp_seed = " << seed << std::endl;
  //     std::cout << "TEST FAILED" << std::endl;
  //     assert(false);
  //   }
  // }
};

void full_run() {
  // Define AIE CGRA running a program "prog" on all the tiles of a VC1902
  // acap::aie::device<acap::aie::layout::vc1902> aie;
  acap::aie::device<acap::aie::layout::size<4, 4>> aie;
  // Run up to completion of all the tile programs
  aie.run<prog, tile_memory>();
}

int main(int argc, char **argv) {
  // std::cout << seed0 << std::endl;
  // while (true) {
  //   seed = seed_dist(mt0);
  //   mt.seed(seed);
  //   for (int i = 0; i < count; i++) {
  //     global_in0[i] = data_dist(mt);
  //     global_in1[i] = data_dist(mt);
  //   }

    full_run();

  //   iter++;
  //   std::cout << "\riteration: " << iter << std::endl;
  // }
  return 0;
}
