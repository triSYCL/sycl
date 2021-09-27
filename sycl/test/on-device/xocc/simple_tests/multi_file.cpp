
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DFILE1 -c %s -o %s1.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -c %s -o %s2.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s1.o %s2.o -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#ifdef FILE1

#include <CL/sycl.hpp>

void test();

int main() {
  test();
  cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);
  cl::sycl::queue Queue{sycl::accelerator_selector{}};
  cl::sycl::range<1> NumOfWorkItems{Buffer.size()};
  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only);
    cgh.single_task<class Test1>([=] { Accessor[0] = 0; });
  });
}
#else

#include <CL/sycl.hpp>

void test() {
  cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);
  cl::sycl::queue Queue{sycl::accelerator_selector{}};
  cl::sycl::range<1> NumOfWorkItems{Buffer.size()};
  Queue.submit([&](cl::sycl::handler &cgh) {
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only);
    cgh.single_task<class Test2>(
        [=] { Accessor[0] = 0; });
  });
}
#endif
