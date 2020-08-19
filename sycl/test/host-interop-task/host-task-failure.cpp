// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t BUFFER_SIZE = 1024;

template <typename T>
class Modifier;

template <typename T>
class Init;

template <typename DataT>
void copy(buffer<DataT, 1> &Src, buffer<DataT, 1> &Dst, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto SrcA = Src.template get_access<mode::read>(CGH);
    auto DstA = Dst.template get_access<mode::write>(CGH);

    CGH.codeplay_host_task([=]() {
      for (size_t Idx = 0; Idx < SrcA.get_count(); ++Idx)
        DstA[Idx] = SrcA[Idx];
    });
  });
}

template <typename DataT>
void init(buffer<DataT, 1> &B1, buffer<DataT, 1> &B2, queue &Q) {
  Q.submit([&](handler &CGH) {
    auto Acc1 = B1.template get_access<mode::write>(CGH);
    auto Acc2 = B2.template get_access<mode::write>(CGH);

    CGH.parallel_for<Init<DataT>>(BUFFER_SIZE, [=](item<1> Id) {
      Acc1[Id] = -1;
      Acc2[Id] = -2;
    });
  });
}

void test() {
  queue Q;
  buffer<int, 1> Buffer1{BUFFER_SIZE};
  buffer<int, 1> Buffer2{BUFFER_SIZE};

  init<int>(Buffer1, Buffer2, Q);

  copy(Buffer1, Buffer2, Q);
}

int main() {
  test();
  return 0;
}
