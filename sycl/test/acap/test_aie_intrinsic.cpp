// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#include <sycl/sycl.hpp>
#include <cstring>
#include <iostream>

using namespace sycl::vendor::xilinx;

template <typename AIE, int X, int Y> struct prog : acap::aie::tile<AIE, X, Y> {
  using t = acap::aie::tile<AIE, X, Y>;
  static constexpr uint32_t array_size = 8;

  bool prerun() {
    // t::mem_write(acap::aie::xaie::aiev1::args_start, 8);
    return 1;
  }

  /// The run member function is defined as the tile program
  unsigned sycl_arg;
  void run() {
    sycl_arg = acap_intr::get_coreid();
  }

  void postrun() {
      // std::cout << "result " << t::mem_read(acap::aie::xaie::aiev1::args_start) << std::endl;
  }
};

int main() {
  // Define AIE CGRA running a program "prog" on all the tiles of a VC1902
  acap::aie::device<acap::aie::layout::size<1, 1>> aie;
  // Run up to completion of all the tile programs
  aie.run<prog>();
  return 0;
}
