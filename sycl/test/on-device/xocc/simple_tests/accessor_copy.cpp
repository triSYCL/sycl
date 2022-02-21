// REQUIRES: xocc
// REQUIRES: spir

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test is testing 2D and 3D D2D copy using the handler it's heavily based
// on the handler_mem_op.cpp test inside basic_tests/handler. In this case we're
// just unit testing a subset of the tests that cause some problems for XRT
// rather than the full comprehensive list contained inside handler_mem_op.cpp.
// \TODO maybe throw in some 1D D2D tests
#include <iostream>
#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class noop;

template<typename T, int xSize, int ySize>
void copy_test_2d(queue q) {
  T Data[xSize][ySize] = {{0}};
  T Values[xSize][ySize] = {{0}};
  for (size_t I = 0; I < xSize; ++I)
    for (size_t J = 0; J < ySize; ++J)
      Data[I][J] = I + J * xSize;

  {
    buffer<T, 2> bufferFrom((T *)Data, range<2>(xSize, ySize));
    buffer<T, 2> bufferTo((T *)Values, range<2>(xSize, ySize));

    q.submit([&](handler &cgh) {
      accessor<T, 2, access::mode::read, access::target::global_buffer>
          accessorFrom(bufferFrom, cgh, range<2>(xSize, ySize));
      accessor<T, 2, access::mode::write, access::target::global_buffer>
          accessorTo(bufferTo, cgh, range<2>(xSize, ySize));
      cgh.copy(accessorFrom, accessorTo);
    });

    q.wait();
  }

  for (size_t I = 0; I < xSize; ++I)
    for (size_t J = 0; J < ySize; ++J)
      assert(Data[I][J] == Values[I][J]);
}

template<typename T, int xSize, int ySize, int zSize>
void copy_test_3d(queue q) {
  T Data[xSize][ySize][zSize] = {{{0}}};
  T Values[xSize][ySize][zSize] = {{{0}}};
  for (size_t I = 0; I < xSize; ++I)
    for (size_t J = 0; J < ySize; ++J)
      for (size_t K = 0; K < zSize; ++K)
      Data[I][J][K] = I + J + K * xSize;

  {
    buffer<T, 3> bufferFrom((T *)Data, range<3>(xSize, ySize, zSize));
    buffer<T, 3> bufferTo((T *)Values, range<3>(xSize, ySize, zSize));

    q.submit([&](handler &cgh) {

      accessor<T, 3, access::mode::read, access::target::global_buffer>
          accessorFrom(bufferFrom, cgh, range<3>(xSize, ySize, zSize));
      accessor<T, 3, access::mode::write, access::target::global_buffer>
          accessorTo(bufferTo, cgh, range<3>(xSize, ySize, zSize));

      cgh.copy(accessorFrom, accessorTo);
    });

    q.wait();
  }

  for (size_t I = 0; I < xSize; ++I)
    for (size_t J = 0; J < ySize; ++J)
      for (size_t K = 0; K < zSize; ++K)
      assert(Data[I][J][K] == Values[I][J][K]);
}

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q{selector};

  // This is a not so smart work around for avoiding cl_mem buffer errors from
  // XRT, until we have lazy buffer generation or early program loading in place
  // for the SYCL runtime.
  // The problem this works around is that buffers can be created but they are
  // not properly mapped to a device inside XRT until a program binary is loaded
  // as it things there is no active device until a binary is loaded. This means
  // things like OpenCL copy calls etc. don't work and are prone to throw
  // runtime errors. What this does is kickstart the runtime into thinking we
  // have an active device to map buffers to, which allows the appropriate buffer
  // mapping to device and OpenCL calls to work.
  {
    const size_t Size = 20;
    int Data[Size] = {0};
    // This showcases another issue we have just now, all Kernels HAVE to have
    // a minimum of 1 accessor or the xocc compiler will complain as there is no
    // buffer associated with m_axi_gmem
    buffer<int, 2> idc((int *)Data, range<2>(Size, Size));
    q.submit([&](handler &cgh) {
      accessor<int, 2, access::mode::write, access::target::global_buffer>
          accessorFrom(idc, cgh, range<2>(Size, Size));
      cgh.single_task<noop>([=](){
        accessorFrom[0][0] = 1;
      });
    });
    q.wait();
  }

  std::cout << "testing 2d 20x20 \n";
  copy_test_2d<int, 20, 20>(q);
  std::cout << "testing 2d 20x40 \n";
  copy_test_2d<int, 20, 40>(q);
  std::cout << "testing 2d 20x25 \n";
  copy_test_2d<int, 20, 25>(q);
  std::cout << "testing 3d 20x20x20 \n";
  copy_test_3d<int, 20,20,20>(q);
  std::cout << "testing 3d 40x20x60 \n";
  copy_test_3d<int, 40,20,60>(q);
  std::cout << "testing 3d 90x50x100 \n";
  copy_test_3d<int, 90,50,100>(q);

  return 0;
}
