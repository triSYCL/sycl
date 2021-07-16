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

#include "CL/sycl/detail/defines.hpp"
#include "CL/sycl/xilinx/fpga/ssdm_inst.hpp"
#include <cstdint>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl::xilinx {

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

   \param[in] f is a function with an innermost loop to be executed in a
    pipeline way.
*/
template <int II = -1, bool rewind = false,
          PipelineStyle pipelineType = PipelineStyle::stall>
struct Pipeline {
  template <typename T>
  __SYCL_DEVICE_ANNOTATE("xilinx_pipeline", II, rewind, pipelineType)
  __SYCL_ALWAYS_INLINE static void decorate(T &&functor) {
    std::forward<T>(functor)();
  }
};

using NoPipeline = Pipeline<0, false>; 

/** Execute loops in a pipelined manner

    A loop with pipeline mode processes a new input every clock
    cycle. This allows the operations of different iterations of the
    loop to be executed in a concurrent manner to reduce latency.

   \tparam rewind if set to true, the pipelined will not be flushed
   between two complete loop executions

   \tparam II the desired Initiation Interval for the loop.
   Special values : -1 for default, 0 for pipeline desactivation.

   \tparam pipelineType which pipeline style to use

   \param[in] f is a function with an innermost loop to be executed in a
    pipeline way.
*/
template <int II = -1, bool rewind = false,
          PipelineStyle pipelineType = PipelineStyle::stall, typename T>
[[ deprecated ("Use Pipeline::decorate instead") ]] __SYCL_ALWAYS_INLINE void pipeline(T &&functor) {
  Pipeline<II, rewind, pipelineType>::decorate(std::forward<T>(functor));
}

template <typename T> [[ deprecated ("Use NoPipeline::decorate instead") ]] __SYCL_ALWAYS_INLINE void no_pipeline(T &&functor) {
  NoPipeline::decorate(std::forward<T>(functor));
}
} // namespace sycl::xilinx
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
