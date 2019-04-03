/*
   This test aims to check if the vector class and related built-ins are still
   broken or if they function correctly (once they work, this should be moved to
   simple_tests or be replaced with a more interesting example utilizing vector
   maths).

   At the time of adding this test the vector overloads of math functions like
   sqrt were causing conflicts when compiling the host side of the code.

   Related to the opened issue: https://github.com/intel/llvm/issues/43
*/

#include <iostream>
#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class vec_math_host_conflict;

// Isn't a problem with pocl or iocl
int main() {
  selector_defines::IntelDeviceSelector iocl;
  queue q{iocl};

  auto e = q.submit([&](handler &cgh) {
      cgh.single_task<vec_math_host_conflict>([=]() {
        float4 f4{2.0};
        auto res = cl::sycl::sqrt(f4);
      });
  });

  q.wait();

  return 0;
}
