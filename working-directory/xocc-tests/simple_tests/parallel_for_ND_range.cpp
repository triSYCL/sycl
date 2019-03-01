#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

/*
  The aim of this test is to check that multidimensional kernels are executing
  appropriately using OpenCL ND range kernels and outputting correct
  information.
*/
using namespace cl::sycl;

class XOCLDeviceSelector : public device_selector {
 public:
   int operator()(const cl::sycl::device &Device) const override {
     using namespace cl::sycl::info;

     const std::string DeviceName = Device.get_info<info::device::name>();
     const std::string DeviceVendor = Device.get_info<info::device::vendor>();

     return (DeviceVendor.find("Xilinx") != std::string::npos) ? 1 : -1;
   }
 };


template <int Dimensions, class kernel_name>
void gen_nd_range(range<Dimensions> k_range) {
  XOCLDeviceSelector xocl;

  queue my_queue{xocl};

  buffer<unsigned int> a(k_range.size());

  my_queue.submit([&](handler &cgh) {
    auto acc = a.get_access<access::mode::write>(cgh);

    cgh.parallel_for<kernel_name>(k_range, [=](item<Dimensions> index) {
            unsigned int range = index.get_range()[0];
            for (size_t i = 1; i < Dimensions; ++i)
              range *= index.get_range()[i];

            acc[index.get_linear_id()] = index.get_linear_id() + range;
        });
  });

  auto acc_r = a.get_access<access::mode::read>();

  for (unsigned int i = 0; i < k_range.size(); ++i) {
      // std::cout << acc_r[i] << " == " << k_range.size() + i << std::endl;
      assert(acc_r[i] == k_range.size() + i &&
        "incorrect result acc_r[i] != k_range.size() + i");
  }

  my_queue.wait();
}

int main(int argc, char *argv[]) {
  gen_nd_range<2, class add>({10, 10}); // change the name later..

  return 0;
}
