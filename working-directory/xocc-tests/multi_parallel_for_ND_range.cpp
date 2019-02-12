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
void gen_nd_range(range<Dimensions> k_range, queue my_queue) {
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
    //  std::cout << acc_r[i] << " == " << k_range.size() + i << std::endl;
      assert(acc_r[i] == k_range.size() + i &&
        "incorrect result acc_r[i] != k_range.size() + i");
  }

  my_queue.wait();
}


/*
  Kernel Names:
  par_1d
  par_2d_square
  par_2d_rect
  par_3d_square
  par_3d_rect
*/
// This test does not deal with duplicate kernel names, it was to test the
// ability to extract all the kernels from a file that contained multiple
// kernels in the one translation unit when using xocc (xpirbc consumption path
// doesn't allow you to pass -k all, each kernel needs to be compiled and linked
// separately before being linked together).
// At the time of this test, unique names for every kernel are a requirement
int main(int argc, char *argv[]) {
  XOCLDeviceSelector xocl;

  queue my_queue{xocl};

  gen_nd_range<1, class par_1d>({10}, my_queue);
  gen_nd_range<2, class par_2d_square>({10, 10}, my_queue);
  gen_nd_range<2, class par_2d_rect>({12, 6}, my_queue);
  gen_nd_range<3, class par_3d_square>({10, 10, 10}, my_queue);
  gen_nd_range<3, class par_3d_rect>({12, 8, 16}, my_queue);

  return 0;
}
