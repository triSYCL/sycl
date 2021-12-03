// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::vendor::xilinx;

template <typename AIE, int X, int Y>
struct prog : acap::aie::tile<AIE, X, Y> {
  void run() {
    int i = X * Y;
  }
};

int main() {
  // Define AIE CGRA with all the tiles of a VC1902
  // acap::aie::device<acap::aie::layout::size<1,1>> aie;
  // acap::aie::device<acap::aie::layout::size<4,2>> aie;
  acap::aie::device<acap::aie::layout::vc1902> aie;
  // Run up to completion prog on all the tiles
  aie.run<prog>();
}
