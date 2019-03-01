#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class are_you_broken;

int main() {
  selector_defines::IntelDeviceSelector iocl;

  queue q {iocl};

  auto e = q.submit([&](handler &cgh) {
    int w = 512;
    cgh.single_task<are_you_broken>([=]() {
      printf("%d \n", w);
    });
  });

  q.wait();
  e.wait();

  return 0;
}
