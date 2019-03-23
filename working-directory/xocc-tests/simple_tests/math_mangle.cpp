/*
  This test is similar to id_mangle.cpp in that it's mostly for testing the
  correct generation of SPIR built-ins when compiling for XOCC. Incorrect
  generation can result in linker errors, run-time ABI errors or incorrect
  output.

  Only really testing against the functions currently in the SYCL math library
  and when this test was created the vector math functionality had some
  problems that Intel plan to fix.
*/

#include <CL/sycl.hpp>
#include <iostream>
#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class math_mangle;

bool CloseEnough(float a, float b)
{
    return fabs(a - b) < 0.000001f;
}

int main(int argc, char* argv[]) {
  selector_defines::XOCLDeviceSelector xocl;

  buffer<float, 1> test_buffer{range<1>{13}};

  queue q { xocl };
  q.submit([&](handler &cgh) {
    auto wb = test_buffer.get_access<access::mode::write>(cgh);

    cgh.single_task<math_mangle>([=]() {
      wb[0] = cl::sycl::sqrt(10.0f);
      wb[1] = cl::sycl::fabs(-10.0f);
      wb[2] = cl::sycl::sin(10.0f);
      wb[3] = cl::sycl::max(10, 12);
      wb[4] = cl::sycl::min(10, 12);
      wb[5] = cl::sycl::floor(10.2f);
      wb[6] = cl::sycl::ceil(10.2f);
      wb[7] = cl::sycl::exp(10.2f);
      wb[8] = cl::sycl::log(10.2f);
      wb[9] = cl::sycl::fmin(10.2f, 10.0f);
      wb[10] = cl::sycl::fmax(10.2f, 10.0f);
      wb[11] = cl::sycl::cos(10.2f);

      // not working, unsure why at the moment, has correct mangling in llvm ir
      // wb[12] = cl::sycl::mad(10.2f, 11.0f, 12.0f)

  });
});

  auto rb = test_buffer.get_access<access::mode::read>();

  printf("sqrt: %f \n", rb[0]);
  assert(CloseEnough(rb[0], 3.162278));

  printf("fabs: %f \n", rb[1]);
  assert(rb[1] == 10.000000);

  printf("sin: %f \n",  rb[2]);
  assert(CloseEnough(rb[2], -0.544021));

  printf("max: %f \n", rb[3]);
  assert(rb[3] == 12.000000);

  printf("min: %f \n", rb[4]);
  assert(rb[4] == 10.000000);

  printf("floor: %f \n", rb[5]);
  assert(rb[5] == 10.000000);

  printf("ceil: %f \n", rb[6]);
  assert(rb[6] == 11.000000);

  printf("exp: %f \n", rb[7]);
  assert(CloseEnough(rb[7], 26903.181641));

  printf("log: %f \n", rb[8]);
  assert(CloseEnough(rb[8], 2.322388));

  printf("fmin: %f \n", rb[9]);
  assert(rb[9] == 10.000000);

  printf("fmax: %f \n", rb[10]);
  assert(CloseEnough(rb[10], 10.200000));

  printf("cos: %f \n", rb[11]);
  assert(CloseEnough(rb[11], -0.714266));

  // printf("mad: %f \n", rb[12]);
  // assert(rb[12] == /*??*/);


  return 0;
}
