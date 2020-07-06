// RUN: true

/*
  Just showcases that you cannot capture a global lambda like this at the moment, perhaps this may change in the future.
*/

#include <CL/sycl.hpp>
#include <iostream>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

auto do_global = [](){};
int main() {
  auto do_local = [=](){};

  selector_defines::CompiledForDeviceSelector selector;
  queue q{selector};

  q.submit([&](handler &cgh) {
    cgh.single_task<class event_wait>([=]() {
      do_local();
      do_global();
    });
  });

  return 0;
}
