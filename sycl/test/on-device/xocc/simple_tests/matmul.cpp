// REQUIRES: spir

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: env SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING=1 %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <random>

using cl::sycl::detail::aligned_allocator;
using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

template<typename A_r, typename A_rw> void matmul(A_rw C, A_r A, A_r B) {
  for (size_t k = 0; k < A.get_range()[0]; k++) {
    for (size_t j = 0; j < A.get_range()[0]; j++) {
      C[{k, j}] = 0;
      for (size_t i = 0; i < A.get_range()[0]; i++) {
        C[{k, j}] += A[{k, i}] * B[{i, j}];
      }
    }
  }
}

int gen_random() {
  static default_random_engine e;
  static uniform_int_distribution<int> dist(0, 10);

  return dist(e);
}

template<typename A> void print(A A_r) {
  for (size_t i = 0; i < std::min(10ul, A_r.get_range()[0]); i++) {
    for (size_t j = 0; j < std::min(10ul, A_r.get_range()[1]); j++) {
      printf("%4d ", A_r[{i, j}]);
    }
    printf("…\n");
  }
  for (int r = 0; r < std::min(10ul, A_r.get_range()[1]); r++) {
    printf("   %s ", "…");
  }
  printf("⋱\n\n");
}

void verify(cl::sycl::accessor<int, 2, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer>
                gold_r,
            cl::sycl::accessor<int, 2, cl::sycl::access::mode::read,
                               cl::sycl::access::target::host_buffer>
                output_r) {
  for (size_t i = 0; i < output_r.get_range()[0]; i++) {
    for (size_t j = 0; j < output_r.get_range()[1]; j++) {
      if (output_r[{i, j}] != gold_r[{i, j}]) {
        printf("Mismatch %zu: gold: %d device: %d\n", i, gold_r[{i, j}],
               output_r[{i, j}]);
        print(output_r);
        printf("TEST FAILED\n\n");
        exit(EXIT_FAILURE);
      }
    }
  }
}

int main() {
  size_t columns = 8;
  size_t rows = columns;

  // Creating SYCL queue
  cl::sycl::queue Queue;

  cl::sycl::buffer<int, 2> A(cl::sycl::range<2>{columns, rows});
  cl::sycl::buffer<int, 2> B(cl::sycl::range<2>{columns, rows});
  cl::sycl::buffer<int, 2> gold(cl::sycl::range<2>{columns, rows});
  cl::sycl::buffer<int, 2> C(cl::sycl::range<2>{columns, rows});
  {
    cl::sycl::accessor A_rw =
        A.get_access<cl::sycl::access::mode::read_write>();
    cl::sycl::accessor B_rw =
        B.get_access<cl::sycl::access::mode::read_write>();
    cl::sycl::accessor G_rw =
        gold.get_access<cl::sycl::access::mode::read_write>();
    for (size_t i = 0; i < A_rw.get_range()[0]; i++)
      for (size_t j = 0; j < A_rw.get_range()[1]; j++) {
        A_rw[{i, j}] = gen_random();
        B_rw[{i, j}] = gen_random();
      }

    printf("A:\n");
    print(A_rw);
    printf("B:\n");
    print(B_rw);
    matmul(G_rw, A_rw, B_rw);
    printf("Gold:\n");
    print(G_rw);
  }

  // Submitting command group(work) to queue
  Queue.submit([&](cl::sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto C_w = C.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto A_r = A.get_access<cl::sycl::access::mode::read>(cgh);
    auto B_r = B.get_access<cl::sycl::access::mode::read>(cgh);
    // Executing kernel
    cgh.single_task<class S>([=] {
      matmul(C_w, A_r, B_r);
    });
  });

  verify(gold.get_access<cl::sycl::access::mode::read>(),
         C.get_access<cl::sycl::access::mode::read>());

  {
    cl::sycl::accessor C_rw =
        C.get_access<cl::sycl::access::mode::read_write>();
    for (size_t k = 0; k < A.get_range()[0]; k++)
      for (size_t j = 0; j < A.get_range()[0]; j++)
        C_rw[{k, j}] = 0;
  }

  // Submitting command group(work) to queue
  Queue.submit([&](cl::sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto C_w = C.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto A_r = A.get_access<cl::sycl::access::mode::read>(cgh);
    auto B_r = B.get_access<cl::sycl::access::mode::read>(cgh);
    // Executing kernel
    cgh.parallel_for<class S2>(A_r.get_range(), [=](cl::sycl::id<2> idx) {
      C_w[{idx[0], idx[1]}] = 0;
      for (size_t i = 0; i < A_r.get_range()[0]; i++) {
        C_w[{idx[0], idx[1]}] += A_r[{idx[0], i}] * B_r[{i, idx[1]}];
      }
    });
  });

  verify(gold.get_access<cl::sycl::access::mode::read>(),
         C.get_access<cl::sycl::access::mode::read>());

  printf("TEST PASSED\n\n");

  return EXIT_SUCCESS;
}
