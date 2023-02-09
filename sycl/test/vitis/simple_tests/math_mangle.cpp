// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*
  This test is similar to id_mangle.cpp in that it's mostly for testing the
  correct generation of SPIR built-ins when compiling for Vitis HLS. Incorrect
  generation can result in linker errors, run-time ABI errors or incorrect
  output.

  Only really testing against the functions currently in the SYCL math library
  and when this test was created the vector math functionality had some
  problems that Intel plan to fix.
*/

#include <sycl/sycl.hpp>
#include <iostream>


using namespace sycl;

class math_mangle;

bool CloseEnough(float a, float b)
{
    return fabs(a - b) < 0.000001f;
}

int main(int argc, char* argv[]) {
  queue q;
  buffer<float, 1> test_buffer{range<1>{13}};

  q.submit([&](handler &cgh) {
    auto wb = test_buffer.get_access<access::mode::write>(cgh);

    cgh.single_task<math_mangle>([=] {
      wb[0] = sycl::sqrt(10.0f);
      wb[1] = sycl::fabs(-10.0f);
      wb[2] = sycl::sin(10.0f);
      wb[3] = sycl::max(10, 12);
      wb[4] = sycl::min(10, 12);
      wb[5] = sycl::floor(10.2f);
      wb[6] = sycl::ceil(10.2f);
      wb[7] = sycl::exp(10.2f);
      wb[8] = sycl::log(10.2f);
      wb[9] = sycl::fmin(10.2f, 10.0f);
      wb[10] = sycl::fmax(10.2f, 10.0f);
      wb[11] = sycl::cos(10.2f);

      // not working, unsure why at the moment, has correct mangling in llvm ir
      // wb[12] = sycl::mad(10.2f, 11.0f, 12.0f)

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
