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

    // __SYCL_XILINX_ONLY__ is a macro definition created by the Clang compiler
    // when fsycl_xocc_device is used, so just making a selector to choose based
    // on this compiler definition to avoid having to repeat code in all of the
    // tests
    class CompiledForDeviceSelector : public device_selector {
     public:
       int operator()(const cl::sycl::device &Device) const override {
#ifdef __SYCL_XILINX_ONLY__
          const std::string DeviceVendor = Device.get_info<info::device::vendor>();
          return (DeviceVendor.find("Xilinx") != std::string::npos) ? 1 : -1;
#else
         const std::string DeviceVendor = Device.get_info<info::device::vendor>();
         return (DeviceVendor.find("Intel(R) Corporation") != std::string::npos) ? 1 : -1;
#endif
       }
     };
}
