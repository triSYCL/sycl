// REQUIRES: xocc && has_secondary_cuda

// RUN: %clangxx -fsycl -fsycl-libspirv-path=%llvm_build_lib_dir./clc/libspirv-nvptx64--nvidiacl.bc -fsycl-targets=nvptx64-nvidia-cuda-sycldevice,%sycl_triple %s -o %t.out  -###
// RUN: %clangxx -fsycl -fsycl-libspirv-path=%llvm_build_lib_dir./clc/libspirv-nvptx64--nvidiacl.bc -fsycl-targets=%sycl_triple,nvptx64-nvidia-cuda-sycldevice %s -o %t.out

// RUN: %ACC_RUN_PLACEHOLDER env --unset=SYCL_DEVICE_FILTER %t.out

#include <CL/sycl.hpp>

constexpr unsigned int size = 12;

template<typename S> struct FillBuffer {};

template<typename Selector> void test_device(Selector s) {
  // Creating buffer of size ints to be used inside the kernel code
  sycl::buffer<int> Buffer{size};

  // Creating SYCL queue
  sycl::queue Queue{s};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.single_task<FillBuffer<Selector>>([=]() {
                                             for (std::size_t i = 0 ; i < size ; ++i) {
                                                 Accessor[i] = i;
                                             }
                                           });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  const auto HostAccessor = Buffer.get_access<sycl::access::mode::read>();

  // Check the results
  bool MismatchFound = false;
  for (std::size_t I = 0; I < size; ++I) {
    if (HostAccessor[I] != I) {
      std::cerr << "The result is incorrect for element: " << I
                << ", expected: " << I << ", got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << __PRETTY_FUNCTION__ << ": The results are correct!"
              << std::endl;
  }
}

int main() {
  test_device(sycl::host_selector{});
  test_device(sycl::accelerator_selector{});
  test_device(sycl::gpu_selector{});
}
