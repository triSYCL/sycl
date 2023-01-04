// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple -DFILE1 -c %s -o %t.dir/o1.o
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple -c %s -o %t.dir/o2.o
// RUN: %run_if_not_cpu %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %t.dir/o1.o %t.dir/o2.o -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#ifdef FILE1

#include <sycl/sycl.hpp>

void test();

int main() {
  test();
  sycl::buffer<sycl::cl_int, 1> Buffer(4);
  sycl::queue Queue;
  sycl::range<1> NumOfWorkItems{Buffer.size()};
  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only);
    cgh.single_task<class Test1>([=] { Accessor[0] = 0; });
  });
}
#else

#include <sycl/sycl.hpp>

void test() {
  sycl::buffer<sycl::cl_int, 1> Buffer(4);
  sycl::queue Queue;
  sycl::range<1> NumOfWorkItems{Buffer.size()};
  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor(Buffer, cgh, sycl::write_only);
    cgh.single_task<class Test2>(
        [=] { Accessor[0] = 0; });
  });
}
#endif
