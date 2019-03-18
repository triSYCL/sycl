#include <iostream>
#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class host_conflict;

// Isn't a problem with pocl or iocl
int main() {
  selector_defines::IntelDeviceSelector iocl;
  queue q{iocl};

  auto e = q.submit([&](handler &cgh) {
      cgh.single_task<host_conflict>([=]() {
        float4 f4{2.0};
        auto res = cl::sycl::sqrt(f4);
      });
  });

  q.wait();

  return 0;
}
