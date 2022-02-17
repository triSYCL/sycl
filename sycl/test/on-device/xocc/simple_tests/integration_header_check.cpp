// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out


/*
  The main point of the test is to check if you can name SYCL kernels in
  certain ways without the compiler or run-time breaking due to an
  incorrectly generated integration header.

  This test is similar to sycl/test/regression/kernel_name_class.cpp

  But started as an executable variation of integration_header.cpp from the
  CodeGenSYCL tests.

*/

#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class kernel_1;

namespace second_namespace {
template <typename T = int>
class second_kernel;
}

template <int a, typename T1, typename T2>
class third_kernel;

struct x {};
template <typename T>
struct point {};

namespace template_arg_ns {
  template <int DimX>
  struct namespaced_arg {};
}

template <typename ...Ts>
class fourth_kernel;

namespace nm1 {
  namespace nm2 {

    template <int X> class fifth_kernel {};
  } // namespace nm2

template <typename... Ts> class sixth_kernel;
}

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};

  buffer<int> ob(range<1>{1});

  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);

    cgh.single_task<kernel_1>([=]() {
      wb[0] = 1;
    });
  });

  {
    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 1 && "kernel execution or assignment error");
    printf("%d \n", rb[0]);
  }

  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);

    cgh.single_task<class kernel_2>([=]() {
      wb[0] += 2;
    });
  });

  {
    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 3 && "kernel execution or assignment error");
    printf("%d \n", rb[0]);
  }

  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);

    cgh.single_task<second_namespace::second_kernel<char>>([=]() {
      wb[0] += 3;
    });
  });

  {
    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 6 && "kernel execution or assignment error");
    printf("%d \n", rb[0]);
  }

  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);

    // note: in the integration header specialization of this kernel it removes
    // the keyword struct from the struct X declaration, it works as it by
    // default re-declares it at the beginning of the header, is this ideal
    // behavior though?
    cgh.single_task<third_kernel<1, int,point<struct X>>>([=]() {
      wb[0] += 4;
    });
  });

  {
    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 10 && "kernel execution or assignment error");
    printf("%d \n", rb[0]);
  }

  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);
    cgh.single_task<fourth_kernel<template_arg_ns::namespaced_arg<1>>>([=]() {
      wb[0] += 5;
    });
  });

  {
    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 15 && "kernel execution or assignment error");
    printf("%d \n", rb[0]);
  }

  q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);
    cgh.single_task<nm1::sixth_kernel<nm1::nm2::fifth_kernel<10>>>([=]() {
      wb[0] += 6;
    });
  });

  {
    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 21 && "kernel execution or assignment error");
    printf("%d \n", rb[0]);
  }

  return 0;
}
