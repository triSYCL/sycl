// RUN: true

/*
  The aim of this test is to check that local memory compiles and works
  reasonably! And.... it doesn't! At the moment.

  Currently Local Memory appears to be allocated as Global Memory, this appear
  to happen despite the correct address space qualification.

  This requires further discussion with the HLS team, but it appears that
  passing a local (correctly qualified with the appropriate address space)
  pointer to the OpenCL kernel function may not be enough.

   HLS in it's C++ form represents local memory as a locally scoped "stack"
   array that will get allocated to BRAM. So perhaps this is the required route
   to go, which would mean we need to generate a pass that will transform
   local address space decorated pointers to a local array.. or have some
   intermediate representation for the local data. Not a simple task!

*/

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>


using namespace cl::sycl;

constexpr size_t sz = 4;

class simple_local;

int main() {
  queue q;

  buffer<unsigned int> a(sz);

  // need to actually implement a local barrier for SPIR and remodel this
  // example to truly tell if this is appropriately using local memory.
  // This test only really funcitonally shows that it's not using global memory
  // in place of local memory...but it could also be implemented via private..
  q.submit([&](handler &cgh) {
    auto acc_g = a.get_access<access::mode::write>(cgh);
    accessor<int, 1, access::mode::read_write, access::target::local>
        local_mem(range<1>(sz / 2), cgh);

   cgh.parallel_for<simple_local>(nd_range<1>(range<1>(sz), range<1>(sz / 2)),
      [=](nd_item<1> item) {
        local_mem[item.get_local_id()[0]] += item.get_local_id()[0];
        acc_g[item.get_global_id()[0]] = local_mem[item.get_local_id()[0]];
      });
  });

  q.wait();

  auto acc_r = a.get_access<access::mode::read>();

  for (unsigned int i = 0; i < sz; ++i) {
    std::cout << acc_r[i] << "\n";
  }

  return 0;
}
