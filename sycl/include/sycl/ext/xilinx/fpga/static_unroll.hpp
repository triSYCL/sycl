//==- parallel_invoke.hpp --- SYCL templated loop unrolling       -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/defines.hpp"

#ifndef SYCL_XILINX_FPGA_PARALLEL_INVOKE_HPP
#define SYCL_XILINX_FPGA_PARALLEL_INVOKE_HPP

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {

namespace ext::xilinx {
namespace detail {

    
template <int firstStep, int firstOutStep, int inc>
void inline normalized_parallel_invoke(auto &loop_step, auto &&loop_condition,
                                       int first, int bound) {
  if constexpr (firstOutStep - firstStep <= 1) {
    // Actual call step
    constexpr int localInc = firstStep * inc;
    int callIdx = first + localInc;
    if (loop_condition(callIdx, bound)) {
      loop_step(callIdx);
    }
  } else {
    // Subdivide call sequence
    constexpr int midpoint = (firstStep + firstOutStep) / 2;
    normalized_parallel_invoke<firstStep, midpoint, inc>(
        loop_step,
        loop_condition, first, bound);

    normalized_parallel_invoke<midpoint, firstOutStep, inc>(
        loop_step,
        loop_condition, first, bound);
  }
}

template <int firstNormalizedStep, int firstNormalizedOutStep, int offset,
          int inc>
void inline normalized_full_parallel_invoke(auto &loop_step) {
  if constexpr (firstNormalizedOutStep - firstNormalizedStep <= 1) {
    // Actual call step
    constexpr int localInc = firstNormalizedStep * inc;
    int callIdx = offset + localInc;
    loop_step(callIdx);
  } else {
    // Subdivide call sequence
    constexpr int midpoint = (firstNormalizedStep + firstNormalizedOutStep) / 2;
    normalized_full_parallel_invoke<firstNormalizedStep, midpoint, offset, inc>(
        loop_step);

    normalized_full_parallel_invoke<midpoint, firstNormalizedOutStep, offset,
                                    inc>(loop_step);
  }
}
} // namespace detail

template <int unrollFactor = 1, int increment = 1>
inline void parallel_invoke(auto &loop_step, auto &loop_condition, int start,
                            int bound) {

  static_assert(unrollFactor >= 1, "Parallel invoke requires a strictly "
                                   "positive unfold factor. For full "
                                   "parallel unrolling use "
                                   "full_parallel_invoke");
  static_assert(increment != 0, "parallel_invoke increment cannot be zero");
  constexpr int newIncrement = unrollFactor * increment;
  for (int i = start;
       loop_condition(i, bound);
       i += newIncrement)
    detail::normalized_parallel_invoke<0, unrollFactor, increment>(
        loop_step,
        loop_condition, i, bound);
}

template <int startIdx, int nbSteps, int increment = 1>
inline void full_parallel_invoke(auto &loop_step) {
  static_assert(nbSteps > 0, "Trying to unroll a loop of 0 steps");
  detail::normalized_full_parallel_invoke<0, nbSteps, startIdx, increment>(
      loop_step);
}
} // namespace ext::xilinx
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
