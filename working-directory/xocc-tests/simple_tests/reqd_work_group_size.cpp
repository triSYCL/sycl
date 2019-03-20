/*
  This test is intended to check the LLVM IR output for correctness rather than
  be executed. The reqd_work_group_size values are nonsensical for a single_task
*/

#include <CL/sycl.hpp>
#include <iostream>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

// --------------------

// Possible TODO:
// When the SYCL compiler implementation supports Functors it might be
// interesting to come back and test an interface like this for its feasibility:
/*
template <typename F, typename... Props>
struct kernel_decorator {
    kernel_decorator(F&& functor) :
      kernel(functor)  {
    }

    void operator()() {
      kernel();
    }

    F kernel;
};

template <typename... Props, typename F>
auto kernel_functor(F&& functor) {
  return kernel_decorator<F, Props...>(std::forward<F>(functor));
};
*/

// defined like this it requires at least 1 template parameter for the current
// single_task interface.
template <typename... Props>
class reqd_work_group_size_test;

template <typename... Props>
class reqd_work_group_size_test2;

template <typename... Props>
class reqd_work_group_size_test3;

// Just using this to make sure the llvm pass doesn't accidentally pick up the
// incorrect thing
namespace cl::sycl::xilinx {
  template <int DimX>
  struct conflict_test {};
}

int main() {
  selector_defines::XOCLDeviceSelector xocc;
  queue q { xocc };

  q.submit([&](handler &cgh) {
      cgh.single_task< reqd_work_group_size_test<
                          xilinx::reqd_work_group_size<16,16,16>,
                          xilinx::conflict_test<30>> >(
        [=]() {}
    );
  });

  q.wait();

  q.submit([&](handler &cgh) {
      cgh.single_task< reqd_work_group_size_test2<
                          xilinx::reqd_work_group_size<1,1,1>,
                          xilinx::reqd_work_group_size<2,2,2>> >(
        [=]() {}
    );
  });

  q.wait();

  q.submit([&](handler &cgh) {
      cgh.single_task< reqd_work_group_size_test3<
                          xilinx::reqd_work_group_size<4,4,4>> >(
        [=]() {}
    );
  });

  q.wait();

  return 0;
}
