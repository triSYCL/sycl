// RUN: true
#include <CL/sycl.hpp>
#include "../utilities/device_selectors.hpp"

// Test case that will trigger an ICE in XOCC's aggressive dead code elimination
// pass. Seems to be address space cast related based on the Accessor[Index]
// trying to access some "external" struct data/not the same address space
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
  const size_t Size = 10;
  point<int> Data[Size] = {0};

  {
    auto Buffer =
        buffer<point<int>, 1>(Data, range<1>(Size), {property::buffer::use_host_ptr()});
    Buffer.set_final_data(nullptr);
    selector_defines::CompiledForDeviceSelector selector;
    queue Queue {selector};
    Queue.submit([&](handler &Cgh) {
      accessor<point<int>, 1, access::mode::write, access::target::global_buffer>
          Accessor(Buffer, Cgh, range<1>(Size));
              Cgh.parallel_for<ice_kernel>(range<1>{Size},
                                         [=](id<1> Index) {
                Accessor[Index] = Index.get(0); // XOCC ICED
       });
    });
  }

  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == I);
  }

  return 0;
}
