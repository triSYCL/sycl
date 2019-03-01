/*
   A simple typical FPGA-like kernel adding 2 vectors
*/
#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace cl::sycl;

constexpr size_t N = 300;
using Type = int;

class XOCLDeviceSelector : public device_selector {
 public:
   int operator()(const cl::sycl::device &Device) const override {
     using namespace cl::sycl::info;

     const std::string DeviceName = Device.get_info<info::device::name>();
     const std::string DeviceVendor = Device.get_info<info::device::vendor>();

     return (DeviceVendor.find("Xilinx") != std::string::npos) ? 1 : -1;
   }
 };

class POCLDeviceSelector : public device_selector {
 public:
   int operator()(const cl::sycl::device &Device) const override {
     using namespace cl::sycl::info;

     const std::string DeviceName = Device.get_info<info::device::name>();
     const std::string DeviceVendor = Device.get_info<info::device::vendor>();

     return (DeviceVendor.find("GenuineIntel") != std::string::npos) ? 1 : -1;
   }
 };

 class IntelDeviceSelector : public device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;

      const std::string DeviceName = Device.get_info<info::device::name>();
      const std::string DeviceVendor = Device.get_info<info::device::vendor>();

      return (DeviceVendor.find("Intel(R) Corporation") != std::string::npos) ? 1 : -1;
    }
  };

int main(int argc, char *argv[]) {
  // 1) Will I need to implement a Xilinx Scheduler, or something that will
  // force it to use the xilinx SW_EMU?
  // 2) If I have to do split compilation rather than direct to binary compilation
  // is it actually worth it? Would it be better to just try to compile direct to binary?

  buffer<Type> a { N };
  buffer<Type> b { N };
  buffer<Type> c { N };

  {
    auto a_b = b.get_access<access::mode::discard_write>();
    // Initialize buffer with increasing numbers starting at 0
    std::iota(&a_b[0], &a_b[a_b.get_count()], 0);
  }

  {
    auto a_c = c.get_access<access::mode::discard_write>();
    // Initialize buffer with increasing numbers starting at 5
    std::iota(&a_c[0], &a_c[a_c.get_count()], 5);
  }

  POCLDeviceSelector pocl;
  XOCLDeviceSelector xocl;

  queue q { xocl };

 // 1) Retrives XOCL device/platforms from XRT runtime
 // 2) cl::sycl::device::get_devices()
 // 3) cl::sycl::platform::get_platforms()
 // 4) cl::sycl::device_selector::select_device()
 // 5) Single stepping until exit from function _ZNK2cl4sycl15device_selector13select_deviceEv
 // 6) ?? Assert Thread 1 "single_task_vec" received signal SIGSEGV, Segmentation fault.

  std::cout << "Queue Device: " << q.get_device().get_info<info::device::name>() << std::endl;
  std::cout << "Queue Device Vendor: " << q.get_device().get_info<info::device::vendor>() << std::endl;

  // Launch a kernel to do the summation
  q.submit([&] (handler &cgh) {
      // Get access to the data
      // no discard_write conversion for buffer_impl::convertSycl2OCLMode
      // auto a_a = a.get_access<access::mode::discard_write>(cgh);

      auto a_a = a.get_access<access::mode::write>(cgh);
      auto a_b = b.get_access<access::mode::read>(cgh);
      auto a_c = c.get_access<access::mode::read>(cgh);

      // A typical FPGA-style pipelined kernel
      cgh.single_task<class add>([=] () {
          // Use an intermediate automatic array
          decltype(a_b)::value_type array[N];
          // This should not generate a call to
          // @llvm.memcpy.p0i8.p1i8.i64 in the SPIR output
          // because it makes argument promotion not working
          for (unsigned int i = 0 ; i < N; ++i)
            array[i] = a_b[i];
          for (unsigned int i = 0 ; i < N; ++i)
            a_a[i] = array[i] + a_c[i];
        });
    });

  // Verify the result
  auto a_a = a.get_access<access::mode::read>();
  for (unsigned int i = 0 ; i < a.get_count(); ++i) {
    // std::cout << a_a[i] << " == " << 5 + 2*i << std::endl;
    assert(a_a[i] == 5 + 2*i && "invalid result from kernel");
  }

  return 0;
}
