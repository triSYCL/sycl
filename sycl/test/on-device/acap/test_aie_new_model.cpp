// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::vendor::xilinx;

int main() {
  // Define an AIE CGRA with all the tiles of a VC1902
  // acap::aie::device<acap::aie::layout::vc1902> d;
  acap::aie::device<acap::aie::layout::size<1, 1>> d;
  //  Submit some work on each tile, which is SYCL sub-device
  d.for_each_tile([](auto& t) {
    /* This will instantiate uniformly the same
       lambda for all the tiles so the tile device compiler is executed
       only once, since each tile has the same code
    */
    t.single_task([&](auto& th) {
      th.log("test1");
      th.log("test2");
      th.log("test3");
    });
  });
  d.wait_all();
}
