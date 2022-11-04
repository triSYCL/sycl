//==- dataflow.hpp --- SYCL Xilinx extension -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes some decorating functions and kernel properties related
/// to dataflow support by Xilinx tools.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_DATAFLOW
#define SYCL_XILINX_FPGA_DATAFLOW

#include <cstdint>
#include <type_traits>
#include <utility>


#include <sycl/detail/defines.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/xilinx/fpga/kernel_properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::xilinx {

/**
  Turn on dataflow optimisation for a loop
*/
struct dataflow {
  template <typename T>
  __SYCL_DEVICE_ANNOTATE("xilinx_dataflow")
  __SYCL_ALWAYS_INLINE dataflow(T functor) {
    functor();
  };
};

auto dataflow_kernel(auto kernel) {
  using kernelType = std::remove_cvref_t<decltype(kernel)>;
  return detail::KernelDecorator<kernelType, decltype(&kernelType::operator()),
                                 decltype("kernel_dataflow"_cstr), 0>{kernel};
}
} // namespace ext::xilinx
}
} // namespace sycl

#endif
