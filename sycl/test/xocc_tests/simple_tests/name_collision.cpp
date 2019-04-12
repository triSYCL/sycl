// RUN: true

#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class add_2;

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};

  q.submit([&] (handler &cgh) {
    cgh.single_task<add_2>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<add_2>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<class add>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<class add>([=] () {
    });
  });

  return 0;
}
