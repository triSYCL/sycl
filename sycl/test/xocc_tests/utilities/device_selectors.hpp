#pragma once

#include <CL/sycl.hpp>


namespace selector_defines {
  using namespace cl::sycl;
  using namespace cl::sycl::info;

  class XOCLDeviceSelector : public device_selector {
   public:
     int operator()(const cl::sycl::device &Device) const override {
       const std::string DeviceVendor = Device.get_info<info::device::vendor>();
       return (DeviceVendor.find("Xilinx") != std::string::npos) ? 1 : -1;
     }
   };

  class POCLDeviceSelector : public device_selector {
   public:
     int operator()(const cl::sycl::device &Device) const override {
       const std::string DeviceVendor = Device.get_info<info::device::vendor>();
       return (DeviceVendor.find("GenuineIntel") != std::string::npos) ? 1 : -1;
     }
   };

   class IntelDeviceSelector : public device_selector {
    public:
      int operator()(const cl::sycl::device &Device) const override {
        const std::string DeviceVendor = Device.get_info<info::device::vendor>();
        return (DeviceVendor.find("Intel(R) Corporation") != std::string::npos) ? 1 : -1;
      }
    };
}
