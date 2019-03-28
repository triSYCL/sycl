#ifndef SYCL_XILINX_KERNEL_PROPERTIES_HPP
#define SYCL_XILINX_KERNEL_PROPERTIES_HPP

#include <boost/core/demangle.hpp>
#include <iostream>
#include <regex>

namespace cl::sycl::xilinx {
  // TOCONSIDER: the typename component could be variadic
  // instead. But that may introduce more complexity than
  // if every property simply takes a single extra type.
  template <int DimX, int DimY, int DimZ, typename T>
  struct reqd_work_group_size {
    static constexpr int x = DimX;
    static constexpr int y = DimY;
    static constexpr int z = DimZ;
  };

  static std::vector<size_t>
  get_reqd_work_group_size(std::string mangledKernelName) {
    // I don't think there is any thread-safety issues around this static regex
    // but leaving a note just in-case this function breaks for unusual reasons
    static std::regex matchInt {"[0-9]+"};
    std::smatch capture;
    const std::string demangled =
    boost::core::demangle(mangledKernelName.c_str());
    std::vector<size_t> reqd;

    if (std::regex_search(demangled, capture,
        std::regex("cl::sycl::xilinx::reqd_work_group_size<[0-9]+,\\s?[0-9]+,\\s?[0-9]+,"))) {
       std::string s = capture[0];
       std::sregex_token_iterator rend;
       std::sregex_token_iterator a {s.begin(), s.end(), matchInt};

       unsigned i = 0;
       while (a!=rend && i < 3) {
         reqd.push_back(std::stoi(*a++));
        ++i;
      }
    }

    return reqd;
  }
}

#endif // SYCL_XILINX_KERNEL_PROPERTIES_HPP
