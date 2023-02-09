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

/// A memory tile has to inherit from acap::aie::memory<AIE, X, Y>
template <typename AIE, int X, int Y>
struct tile_memory : acap::aie::memory<AIE, X, Y> {
  // void* arr[12];
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
#ifdef __SYCL_DEVICE_ONLY__
    acap::heap::dump_allocator_state();
#endif
    {
      std::vector<std::string> strs;
      strs.push_back("abcd ");
      strs.back() += "efgh ";
      strs.back() += "ijkl0 ";
      strs.push_back("abcd2 ");
      strs.back() += "efgh ";
      strs.back() += "ijkl1 ";
      strs.push_back("abcd3 ");
      strs.back() += "efgh ";
      strs.back() += "ijkl\n";
      std::string str;
      for (auto s : strs)
        str += s;

      str += "abcd ";
      str += "efgh ";
      str += "ijkl0 \n";
      t::log(str.c_str());
      t::log("end\n");
    }
#ifdef __SYCL_DEVICE_ONLY__
    acap::heap::dump_allocator_state();
#endif
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
