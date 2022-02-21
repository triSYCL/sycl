// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <CL/sycl.hpp>
#include "../utilities/device_selectors.hpp"

// A regression test case that used to trigger an ICE in XOCC's aggressive dead
// code elimination pass. It was originally related an address space cast based
// on the Accessor[Index] trying to access some "external" struct data/not in
// the same address space. The problem was fixed by some tweaks inside of the
// accessor class and a transition towards the InferAddressSpaces pass.
//
// This example now simply assigns the index value to each element of the
// strucutre via copy constructor and then checks if the values were properly
// assigned via the overloaded equality operator on the host.
using namespace cl::sycl;

template <typename T>
struct point {
  point(const point &rhs) : x(rhs.x), y(rhs.y) {}
  point(T x, T y) : x(x), y(y) {}
  point(T v) : x(v), y(v) {}
  point() : x(0), y(0) {}
  bool operator==(const T &rhs) { return rhs == x && rhs == y; }
  bool operator==(const point<T> &rhs) { return rhs.x == x && rhs.y == y; }

  T x, y;
};

class ice_kernel;

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue Queue {selector};

  const size_t Size = 10;
  point<int> Data[Size] = {0};

  {
    auto Buffer =
      buffer<point<int>, 1>(Data, range<1>(Size),
        {property::buffer::use_host_ptr()});
    Queue.submit([&](handler &Cgh) {
      accessor<point<int>, 1, access::mode::write, access::target::global_buffer>
          Accessor(Buffer, Cgh, range<1>(Size));
              Cgh.parallel_for<ice_kernel>(range<1>{Size}, [=](id<1> Index) {
                Accessor[Index] = Index.get(0);
       });
    });
  }

  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == I);
  }

  return 0;
}
