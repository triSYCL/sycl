// REQUIRES: vitis
// Pipes are not implemented for the host device

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %run_if_not_cpu %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

struct data {
  int i;
  double d;
  bool operator==(const data &other) const {
    return i == other.i && d == other.d;
  }
};

int main(int argc, char *argv[]) {
  int size = 4;
  sycl::queue q;

  sycl::buffer<data, 1> a(size);
  sycl::buffer<data, 1> b(size);
  sycl::buffer<data, 1> c(size);

  {
    sycl::host_accessor a_a(a);
    for (int i = 0; i < size; i++) {
      a_a[i] = data{i, i * 1.5};
    }
  }

  using PipeA = sycl::ext::intel::pipe<class PipeNameA, data>;

  q.submit([&](handler &cgh) {
    sycl::accessor a_a{a, cgh, sycl::read_only};
    cgh.single_task([=] {
      for (unsigned int i = 0; i < size; ++i) {
        PipeA::write(a_a[i]);
      }
    });
  });
  q.submit([&](handler &cgh) {
    sycl::accessor a_c{c, cgh, sycl::write_only};
    cgh.single_task([=] {
      for (unsigned int i = 0; i < size; ++i) {
        a_c[i] = PipeA::read();
      }
    });
  });

  {
    sycl::host_accessor a_c(c);
      for (int i = 0; i < size; i++) {
        data res = data{i, i * 1.5};
        data val = a_c[i];
        assert(val == res);
      }
  }
}
