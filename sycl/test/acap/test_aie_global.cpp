// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin

#define TRISYCL_DEVICE_ALLOCATOR_DEBUG

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

struct A {
  static constexpr unsigned size = 5;
  const char* str;
  int arr[size];
  A(const char* s) {
    str = s;
    acap::multi_log(str, "\n");
  }
  void access() {}
  ~A() {
    acap::multi_log(str, "\n");
  }
};

A a{"a"};
A b{"b"};
A c{"c"};
A d{"d"};
A e{"e"};
A f{"f"};

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

  void run() {
    t::log("start\n");
    a.access();
    b.access();
    c.access();
    d.access();
    e.access();
    f.access();
    t::log("end\n");
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
