// REQUIRES: xocc

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <numeric>
#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>


using namespace sycl;

void aip_test() {
  sycl::ext::xilinx::partition_ndarray<int, sycl::ext::xilinx::dim<2, 3>, sycl::ext::xilinx::partition::complete<>> array = {{1, 2, -1}, {3, 4, -2}};
  const sycl::ext::xilinx::partition_ndarray<int, sycl::ext::xilinx::dim<2, 3>> carray = array;
  sycl::ext::xilinx::partition_ndarray<int, sycl::ext::xilinx::dim<2, 3>, sycl::ext::xilinx::partition::block<2>> array2;

  static_assert(std::is_same_v<decltype(array.begin()), int(*)[3]>, "");
  static_assert(std::is_same_v<decltype(array.end()), int(*)[3]>, "");
  static_assert(std::is_same_v<decltype(array.begin()[0]), int(&)[3]>, "");
  static_assert(std::is_same_v<decltype(array.begin()[0][1]), int&>, "");
  static_assert(std::is_same_v<decltype(array[0]), int(&)[3]>, "");
  static_assert(std::is_same_v<decltype(array[0][1]), int&>, "");

  static_assert(std::is_same_v<decltype(carray.begin()), const int(*)[3]>, "");
  static_assert(std::is_same_v<decltype(carray.end()), const int(*)[3]>, "");
  static_assert(std::is_same_v<decltype(carray.begin()[0]), const int(&)[3]>, "");
  static_assert(std::is_same_v<decltype(carray.begin()[0][1]), const int&>, "");
  static_assert(std::is_same_v<decltype(carray[0]), const int(&)[3]>, "");
  static_assert(std::is_same_v<decltype(carray[0][1]), const int&>, "");

  assert(array[0][0] == 1);
  assert(array[0][1] == 2);
  assert(array[0][2] == -1);
  assert(array[1][0] == 3);
  assert(array[1][1] == 4);
  assert(array[1][2] == -2);

  int count = 0;
  for (auto& l: array)
    for (auto& e: l)
      e = count++;

  count = 0;
  for (auto& l: array)
    for (auto& e: l)
      assert(e == count++);
  
  assert(array != carray);
  assert(!(array == carray));

  array2 = array;
  array = carray;

  assert(array == carray);
  assert(!(array != carray));

  assert(array[0][0] == 1);
  assert(array[0][1] == 2);
  assert(array[0][2] == -1);
  assert(array[1][0] == 3);
  assert(array[1][1] == 4);
  assert(array[1][2] == -2);
}

int main() {
  aip_test();
  sycl::queue q;
  int N = 12;
  sycl::buffer<int> a { N };
  sycl::buffer<int> b { N };
  sycl::buffer<int> c { N };

  {
    auto a_b = b.get_access<sycl::access::mode::discard_write>();
    // Initialize buffer with increasing numbers starting at 0
    std::iota(&a_b[0], &a_b[a_b.size()], 0);
    auto a_c = c.get_access<sycl::access::mode::discard_write>();
    // Initialize buffer with increasing numbers starting at 0
    std::iota(&a_c[0], &a_c[a_c.size()], 1);
  }

  // Launch a kernel to do the summation
  q.submit([&](sycl::handler &cgh) {
    // Get access to the data
    sycl::accessor a_a{a, cgh, sycl::write_only};
    sycl::accessor a_b{b, cgh, sycl::read_only};
    sycl::accessor a_c{c, cgh, sycl::read_only};

    cgh.single_task<class add>([=] {
      sycl::ext::xilinx::partition_ndarray<
          int, sycl::ext::xilinx::dim<2, 3, 2>, sycl::ext::xilinx::partition::complete<>>
          array;
      int i = 0;
      for (auto& d1 : array)
        for (auto& d2 : d1)
          for (auto& e : d2)
            e = a_b[i++];
      i = 0;
      for (auto &d1 : array)
        for (auto &d2 : d1)
          for (auto &e : d2) {
            a_a[i] = e + a_c[i];
            i++;
          }
    });
  });

  {
    auto a_a = a.get_access<sycl::access::mode::read>();
    auto a_b = b.get_access<sycl::access::mode::read>();
    auto a_c = c.get_access<sycl::access::mode::read>();

    for (unsigned int i = 0 ; i < a.size(); ++i)
      assert(a_a[i] == a_b[i] + a_c[i]);
  }
}
