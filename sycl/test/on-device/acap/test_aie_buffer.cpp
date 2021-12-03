// REQUIRES: acap

// RUN: %acap_clang %s -o %s.bin
// RUN: %add_acap_result %s.bin
// RUN: rm %s.bin



#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::vendor::xilinx;

int main() {
  // Define an AIE CGRA with all the tiles of a VC1902
  acap::aie::device<acap::aie::layout::size<1,1>> d;
  // 1 buffer per tile
  sycl::buffer<int> b[d.x_size][d.y_size];
  // Initialize on the host each buffer with 3 sequential values
  d.for_each_tile_index([&](int x, int y) {
    b[x][y] = { 3 };
    sycl::host_accessor a { b[x][y] };
    std::iota(a.begin(), a.end(), (d.x_size * y + x) * a.get_count());
  });
  //  Submit some work on each tile, which is SYCL sub-device
  d.for_each_tile_index([&](int x, int y) {
    d.tile(x, y).submit([&](auto& cgh) {
      acap::aie::accessor a { b[x][y], cgh };
      cgh.single_task([=] {
        for (auto& e : a) {
          e += 42;
          trisycl::vendor::xilinx::acap::multi_log("elem[]=", e,"\n");
        }
      });
    });
  });
  d.wait(); // implicitly done inside single_task
  // Check the result
  d.for_each_tile_index([&](int x, int y) {
    for (sycl::host_accessor a{b[x][y]};
         auto &&[i, e] : ranges::views::enumerate(a)) {
      int expected_value = (d.x_size * y + x) * a.get_count() + i + 42;
      if (e != expected_value)
        throw "Bad computation";
    }
  });
}
