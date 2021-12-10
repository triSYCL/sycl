// Test that a single SYCL application can target the host CPU,
// a Xilinx FPGA device and a CUDA GPU.

#include <CL/sycl.hpp>

template<typename S>
struct FillBuffer {};

template<typename Selector>
void test_device(Selector s) {
  // Creating buffer of 4 ints to be used inside the kernel code
  sycl::buffer<int> Buffer{4};

  // Creating SYCL queue
  sycl::queue Queue{s};

  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<FillBuffer<Selector>>(
        NumOfWorkItems, [=](sycl::id<1> WIid) {
          // Fill buffer with indexes
          Accessor[WIid] = WIid.get(0);
        });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  const auto HostAccessor = Buffer.get_access<sycl::access::mode::read>();

  // Check the results
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cerr << "The result is incorrect for element: " << I
                << ", expected: " << I << ", got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << __PRETTY_FUNCTION__ << ": The results are correct!" << std::endl;
  }
}

int main() {
  test_device(sycl::host_selector{});
  test_device(sycl::accelerator_selector{});
  test_device(sycl::gpu_selector{});
}
