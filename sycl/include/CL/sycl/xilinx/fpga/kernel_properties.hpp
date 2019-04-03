//==- kernel_properties.hpp --- SYCL Xilinx kernel proprerties       -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  This file contains the kernel property template classes that can be applied
///  to kernel names to apply properties to that kernel.
///  It also contains helper functions oriented around these kernel properties.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_KERNEL_PROPERTIES_HPP
#define SYCL_XILINX_FPGA_KERNEL_PROPERTIES_HPP

#include <boost/core/demangle.hpp>
#include <iostream>
#include <regex>

namespace cl::sycl::xilinx {
  /// This is the reqd_work_group_size property that you can wrap around SYCL
  /// kernel names (defined as classes or structs). It applies the OpenCL
  /// reqd_work_group_size attribute to the kernel that is generated.
  ///
  /// \todo: Consider the typename component could be variadic instead. But that
  /// may introduce more complexity than if every property simply takes a single
  /// extra type.
  template <int DimX, int DimY, int DimZ, typename T>
  struct reqd_work_group_size {
    static constexpr int x = DimX;
    static constexpr int y = DimY;
    static constexpr int z = DimZ;
  };

  /// Retrieves the ReqdWorkGroupSize values from a demangled function name
  /// using regex.
  ///
  /// In SYCL, kernel names are defined by types and in our current
  /// implementation we wrap our SYCL kernel names with properties that are
  /// defined as template types. For example ReqdWorkGroupSize is defined as
  /// one of these when the kernel name is translated from type to kernel name
  /// the information is retained and we can retrieve it in this LLVM pass by
  /// using regex on it.
  static std::vector<size_t>
  get_reqd_work_group_size(std::string mangledKernelName) {
    static const std::regex matchSomeNaturalInteger {R"(\d+)"};
    static const std::regex matchReqdWorkGroupSize {
      R"(cl::sycl::xilinx::reqd_work_group_size<\d+,\s?\d+,\s?\d+,)"
    };

    std::smatch capture;
    const std::string demangled =
      boost::core::demangle(mangledKernelName.c_str());
    std::vector<size_t> reqd;

    if (std::regex_search(demangled, capture, matchReqdWorkGroupSize)) {

       std::string s = capture[0];
       std::sregex_token_iterator workGroupSizes{s.begin(), s.end(),
                                          matchSomeNaturalInteger};
       // only really care about the first 3 values, anymore and the
       // reqd_work_group_size interface is incorrect
       for (unsigned i = 0;
            i < 3 && workGroupSizes != std::sregex_token_iterator{};
            ++i, ++workGroupSizes) {
          reqd.push_back(std::stoi(*workGroupSizes));
       }

       if (reqd.size() != 3)
        throw runtime_error("The reqd_work_group_size properties dimensions are"
                            " not equal to 3");
    }

    return reqd;
  }
}

#endif // SYCL_XILINX_FPGA_KERNEL_PROPERTIES_HPP
