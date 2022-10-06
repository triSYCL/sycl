// This is not yet implemented
// REQUIRES: vitis
// Pipes are not implemented for the host device

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

using PipeA = sycl::ext::intel::pipe<class PipeNameA, int>;

__attribute__((noinline)) void writeA(int i) {
  PipeA::write(i);
}

__attribute__((noinline)) int readA() {
  return PipeA::read();
}

int main(int argc, char *argv[]) {
  int size = 4;
  sycl::queue q;
  sycl::buffer<int, 1> a(size);
  sycl::buffer<int, 1> b(size);
  sycl::buffer<int, 1> c(size);

  {
    sycl::host_accessor a_a(a);
    sycl::host_accessor a_b(b);
    for (int i = 0; i < size; i++) {
      a_a[i] = i;
      a_b[i] = i + 1;
    }
  }

  q.submit([&](handler &cgh) {
    sycl::accessor a_a{a, cgh, sycl::read_only};
    cgh.single_task([=] {
      for (unsigned int i = 0; i < size; ++i) {
        writeA(a_a[i]);
      }
    });
  });
  q.submit([&](handler &cgh) {
    sycl::accessor a_c{c, cgh, sycl::write_only};
    cgh.single_task([=] {
      for (unsigned int i = 0; i < size; ++i) {
        a_c[i] = readA();
      }
    });
  });

  {
    sycl::host_accessor a_c(c);
      for (int i = 0; i < size; i++) {
        int res = i;
        int val = a_c[i];
        assert(val == res);
      }
  }
}
