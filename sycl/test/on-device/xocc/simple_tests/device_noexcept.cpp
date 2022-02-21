// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out

// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out


/*
  This tests that the device is not generating exception related IR like
  personality functions and landing pads.

  A previous side effect of using noexcept on the lambda and invoke function in
  this case was the generation of exception related IR on the device when
  exceptions were not appropriately switched off or handled in the compiler.

  The device should be compiled with no exception related code, exception
  related IR can lead to choking of the LLVM-SPIRV translator or the xocc
  backend in some cases.

  If this test case is broken, the test will most likely ICE and you'll know
  about it the hard-way unfortunately!
*/
#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class exceptions_on_device;

template <typename T>
void invoke(T func) noexcept {
  func();
}

int return_v() noexcept {
  return 1;
}

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};
  buffer<int> ob(range<1>{1});

  q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.single_task<exceptions_on_device>([=]() {
        invoke([&]() noexcept {
            wb[0] += return_v();
          }
        );
      });
  });

  q.wait();

  return 0;
}
