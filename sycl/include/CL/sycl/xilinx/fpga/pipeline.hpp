//==- pipeline.hpp --- SYCL Xilinx extension ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes some decorating functions and kernel properties related
/// to pipeliening support by Xilinx tools.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_PIPELINE
#define SYCL_XILINX_FPGA_PIPELINE

#include <cstdint>
#include <type_traits>
#include <utility>

#include "CL/sycl/detail/defines.hpp"
#include "CL/sycl/detail/property_helper.hpp"

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {
namespace xilinx {

enum struct PipelineStyle : std::uint8_t {
  stall = 0,     ///< Runs only when input data is available otherwise it stalls
                 ///< (default).
  flushable = 1, ///< Runs when input data is available or when there is still
                 ///< data in the pipeline otherwise it stalls.
  free_running =
      2, ///< Runs without stalling even when there is no input data available
};

/** Xilinx pipeline.

  Can be used as a kernel propertie or to decorate a loop

 \tparam rewind if set to true, the pipelined will not be flushed
 between two complete loop executions. This parameter is unused on
 kernel properties.

 \tparam II the desired Initiation Interval for the loop.
 Special values : -1 for default, 0 for pipeline desactivation.

 \tparam pipelineType which pipeline style to use

  \tparam T type of the functor to execute
*/
template <int II = -1, bool rewind = false,
          PipelineStyle pipelineType = PipelineStyle::stall, typename T>
__SYCL_DEVICE_ANNOTATE("xilinx_pipeline", II, rewind, pipelineType)
__SYCL_ALWAYS_INLINE void pipeline(T &&functor) { std::forward<T>(functor)(); }

template <typename T>
__SYCL_DEVICE_ANNOTATE("xilinx_pipeline", 0, false, PipelineStyle::stall)
__SYCL_ALWAYS_INLINE void noPipeline(T &&functor) {
  std::forward<T>(functor)();
}
} // namespace xilinx

namespace xilinx {
template <int II = -1, PipelineStyle pipelineType = PipelineStyle::stall>
auto pipeline_kernel(auto kernel) {
  using namespace detail::literals;
  using kernelType = std::remove_cvref_t<decltype(kernel)>;
  return detail::KernelDecorator<kernelType, decltype(&kernelType::operator()),
                                 decltype("kernel_pipeline"_cstr), pipelineType,
                                 II>{kernel};
}

auto unpipeline_kernel(auto kernel) {
  using namespace detail::literals;
  using kernelType = std::remove_cvref_t<decltype(kernel)>;
  return detail::KernelDecorator<kernelType, decltype(&kernelType::operator()),
                                 decltype("kernel_pipeline"_cstr), 0, 0>{
      kernel};
}
} // namespace xilinx
} // namespace sycl

} // __SYCL_INLINE_NAMESPACE(cl)

#endif
