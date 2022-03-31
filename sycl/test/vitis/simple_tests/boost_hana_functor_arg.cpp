// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <CL/sycl.hpp>

#include <boost/hana.hpp>
#include <iostream>

// This test cases showcases a boost hana / SYCL kernel IR issue it should be
// compiled with: -Xclang -fsycl-allow-func-ptr, otherwise Clang diagnostics
// will diagnose errors
using namespace cl::sycl;

constexpr size_t N = 5;

int main(int argc, char *argv[]){
  queue q;

  buffer<int, 1> a{{N}};

  {
    auto a_w = a.get_access<access::mode::write>();
    for (int i = 0; i < N; ++i) {
      a_w[i] = i;
    }
  }

  auto event = q.submit([&](handler &cgh) {
      auto a_rw = a.get_access<access::mode::read_write>(cgh);

      cgh.single_task<class array_add>([=] {
          // Will not work for hw_emu, this will break compilation
          // The current guess is that this seems to be because of the
          // `const auto i` argument having its address space
          // deduced a little off (maybe it isn't off, it could just be because
          // it requires an address space bitcast) causing the optimization pass
          // to have a difficult time removing it. Which subsequently means only
          // some of the IR is fully inlined which leaves us with some empty
          // function defintions and some weird invocations to those definitions
          //
          // And if the IR isn't fit for consumption for hw_emu, it's unlikely
          // that it's fit for consumption for HW
          boost::hana::int_<N>::times.with_index([&](const auto i) {
             a_rw[i+1] = 6;
          });

          // Works/work around...
          int i = 0;
           boost::hana::int_<N>::times([&] {
             a_rw[i+0] = 6;
             ++i;
         });
     });
  });

  q.wait();

  {
    auto a_r = a.get_access<access::mode::read>();
    for (int i = 0; i < N; ++i)
      std::cout << a_r[i] << "\n";
  }

  return 0;
}
