/* REQUIRES: xocc
   REQUIRES: xocc

   RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
   RUN: %clangxx -fsycl -fsycl-unnamed-lambda -std=c++20 -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out  
   RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out 2>&1 | FileCheck %s
   CHECK: 6 8 10
   CHECK: 352 -128 -44.25 -55.875
*/
#include <cstdint>
#include <functional>
#include <iostream>
#include <list>
#include <set>
#include <span>
#include <vector>

#include <boost/hana.hpp>

#include <sycl/ext/xilinx/fpga.hpp>
#include <sycl/sycl.hpp>

using namespace boost::hana::literals;

/* A generic function taking any number of arguments of any type and
   folding them with a given generic operator */
auto generic_executor(auto op, auto... inputs) {
  // Use a tuple of heterogeneous buffers to wrap the inputs
  auto bufs = boost::hana::make_tuple(
      sycl::buffer{std::begin(inputs), std::end(inputs)}...);

  /* The element-wise computation

     Note that we could use HANA to add some hierarchy in the
     computation (Wallace's tree...) or to sort by type to minimize
     the hardware usage... */
  auto compute = [=](auto args) { return boost::hana::fold_left(args, op); };

  /* Use the range of the first argument as the range
     of the result and computation */
  auto size = bufs[0_c].size();

  // Infer the type of the output from 1 computation on inputs
  using return_value_type =
      decltype(compute(boost::hana::make_tuple(*std::begin(inputs)...)));

  // Create a buffer to return the result
  sycl::buffer<return_value_type> output{size};

  // Submit a command-group to the device
  sycl::queue{}.submit([&](sycl::handler &cgh) {
    // Define the data used as a tuple of read accessors
    auto ka = boost::hana::transform(bufs, [&](auto b) {
      return sycl::accessor{b, cgh, sycl::read_only};
    });
    // Data are produced to a write accessor to the output buffer
    sycl::accessor ko{output, cgh, sycl::write_only, sycl::no_init};

    // Define the kernel
    cgh.single_task([=] {
      for (int i = 0; i < size; ++i)
        sycl::ext::xilinx::pipeline([&] {
          // Pack operands an elemental computation in a tuple
          auto operands =
              boost::hana::transform(ka, [&](auto acc) { return acc[i]; });
          // Assign computation on the operands to the elemental result
          ko[i] = compute(operands);
        });
    });
  });
  // Return the output buffer
  return output;
};

int main() {
  std::vector<std::int8_t> u{1, 2, 3};
  std::vector<std::int16_t> v{5, 6, 7};

  // Do not use std::plus because it forces the same type for both operands
  auto res = generic_executor([](auto x, auto y) { return x + y; }, u, v);
  for (sycl::host_accessor a{res, sycl::read_only};
       auto e : std::span{&a[0], a.size()})
    std::cout << e << ' ';
  std::cout << std::endl;

  // Just for kidding
  std::vector<double> a{1, 2.5, 3.25, 10.125};
  std::set<char> b{5, 6, 7, 2};
  std::list<float> c{-55, 6.5, -7.5, 0};
  auto res2 =
      generic_executor([](auto x, auto y) { return 3 * x - 7 * y; }, a, b, c);
  for (sycl::host_accessor a{res2, sycl::read_only};
       auto e : std::span{&a[0], a.size()})
    std::cout << e << ' ';
  std::cout << std::endl;
}
