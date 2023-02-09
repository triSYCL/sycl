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
  sycl::buffer<int> in1[d.x_size][d.y_size];
  sycl::buffer<int> in2[d.x_size][d.y_size];
  sycl::buffer<int> out[d.x_size][d.y_size];
  // Initialize on the host each buffer with 3 sequential values
  int num = 42;
  d.for_each_tile_index([&](int x, int y) {
    in1[x][y] = { 3 };
    sycl::host_accessor ain1 { in1[x][y] };
    std::iota(ain1.begin(), ain1.end(), (d.x_size * y + x) * ain1.get_count());
    in2[x][y] = { 3 };
    sycl::host_accessor ain2 { in2[x][y] };
    for (auto& e: ain2)
      e = 7;
    out[x][y] = { 3 };
  });
  //  Submit some work on each tile, which is SYCL sub-device
  int coef1 = -10;
  int coef2 = 4;
  int coef3 = 7;
  d.for_each_tile_index([&](int x, int y) {
    d.tile(x, y).submit([&](auto& cgh) {
      acap::aie::accessor ain1 { in1[x][y], cgh };
      acap::aie::accessor ain2 { in2[x][y], cgh };
      acap::aie::accessor aout { out[x][y], cgh };
      cgh.single_task([=] {
        for (int i = 0; i < aout.get_count(); i++) {
          aout[i] = coef1 * ain1[i] + coef2 * ain2[i] + coef3;
          trisycl::vendor::xilinx::acap::multi_log(
              "elem[", i, "] = ", coef1, " * ", ain1[i], " + ", coef2, " * ",
              ain2[i], " + ", coef3, " = ", aout[i], "\n");
        }
        std::abort();
      });
    });
  });
  d.wait();
  // Check the result
  d.for_each_tile_index([&](int x, int y) {
    sycl::host_accessor aout{out[x][y]};
    sycl::host_accessor ain1{in1[x][y]};
    sycl::host_accessor ain2{in2[x][y]};
    for (int i = 0; i < aout.get_count(); i++) {
      int expected_value = coef1 * ain1[i] + coef2 * ain2[i] + coef3;
      if (aout[i] != expected_value)
        throw "Bad computation";
    }
  });
}
