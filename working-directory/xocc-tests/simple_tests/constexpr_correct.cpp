/*
  Testing if constexpr values carry across as expected from the host to the
  device e.g. as a compile time hardcoded value and not by value copy
  (trivially testable at the moment as by value capture and transfer appears
  to be broken).
*/

#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

int add(int v1, int v2) {
  return v1 + v2;
}

int main() {
  selector_defines::IntelDeviceSelector iocl;
  queue q { iocl };

  constexpr int host_to_device = 20;
  int try_capture = 30;

  printf("host host_to_device before: %d \n", host_to_device);
  printf("host try_capture before: %d \n", try_capture);
  printf("host add before: %d \n", add(host_to_device, try_capture));

  buffer<int> ob(range<1>{3});

  q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);

      cgh.single_task<class constexpr_carryover>([=]() {
        wb[0] = host_to_device;
        wb[1] = try_capture;
        wb[2] = add(host_to_device, try_capture);
      });
  });

  q.wait();

  auto rb = ob.get_access<access::mode::read>();

  printf("host host_to_device after: %d \n", rb[0]);
  printf("host try_capture after: %d \n", rb[1]);
  printf("host add after: %d \n", rb[2]);

  return 0;
}
