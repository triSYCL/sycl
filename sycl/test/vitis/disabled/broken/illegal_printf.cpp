// REQUIRES: xocc

// RUN: true
// TODO Move to Sema

#include <CL/sycl.hpp>


/*
  This tests that the device code will not compile a printf (variadic C
  function), which is normal SYCL behaviour. You should use the stream class.

  However, it is noteable that you can turn off this rule via a compiler
  command. The example wil compile, but that doesn't mean the behaviour is
  correct for all devices (e.g. this example will print fine on sw and hw
  emulation for Xilinx FPGA, but not on Intel CPU/GPU devices)
*/

using namespace cl::sycl;

class are_you_broken;

int main() {
  queue q;

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
