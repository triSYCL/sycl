// REQUIRES: vitis
// Pipes are not implemented for the host device

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %run_if_not_cpu %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

int main(int argc, char *argv[]) {
  int size = 4;
  sycl::queue q;
  sycl::buffer<int, 1> a(size);
  sycl::buffer<int, 1> c(size);

  {
    sycl::host_accessor a_a(a);
    for (int i = 0; i < size; i++) {
      a_a[i] = i;
    }
  }

  using PipeA = sycl::ext::intel::pipe<class PipeNameA, int>;
  using PipeB = sycl::ext::intel::pipe<class PipeNameB, int>;
  using PipeC = sycl::ext::intel::pipe<class PipeNameC, int>;

  q.submit([&](handler &cgh) {
    sycl::accessor a_a{a, cgh, sycl::read_only};
    cgh.single_task([=] {
      for (unsigned int i = 0; i < size; ++i) {
        bool choose = (a_a[i] % 2) == 0;
        PipeC::write(choose);
        if (choose)
          PipeA::write(a_a[i]);
        else
          PipeB::write(a_a[i]);
      }
    });
  });
  q.submit([&](handler &cgh) {
    sycl::accessor a_c{c, cgh, sycl::write_only};
    cgh.single_task([=] {
      for (unsigned int i = 0; i < size; ++i) {
        bool choose = PipeC::read();
        if (choose)
          a_c[i] = PipeA::read();
        else
          a_c[i] = PipeB::read();
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
