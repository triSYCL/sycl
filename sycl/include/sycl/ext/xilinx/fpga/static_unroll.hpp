//==- static_unroll.hpp --- SYCL templated loop unrolling       -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/detail/defines.hpp"

#ifndef SYCL_XILINX_FPGA_STATIC_UNROLL_HPP
#define SYCL_XILINX_FPGA_STATIC_UNROLL_HPP

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {

namespace ext::xilinx {
namespace detail {

///
///@brief partial_static_unroll implementation
///
/// Flatten a given number of loop step invocation.
///
/// normalized_partial_static_unroll<0, 4, Inc>(LoopStep, Condition, i, bound)
/// is equivalent to:
///
/// \code{.cpp}
/// if (Condition(i, bound)) LoopStep(i);
/// if (Condition(i + Inc, bound)) LoopStep(i + Inc);
/// if (Condition(i + 2*Inc, bound)) LoopStep(i + 2*Inc);
/// if (Condition(i + 3*Inc, bound)) LoopStep(i + 3*Inc);
/// \endcode
///
///@tparam NormalizedInitStep Normalized unrolled iteration index (used for
///        recursion, for top call will be 0)
///@tparam NormalizedExitStep First value greater than maximal normalized
///        unrolled iteration index (Used for recursion: for top call should be
///        the number of step to explicite)
///@tparam Increment How much to add to the iteration variable between each
///        iteration
///@param LoopStep Computation performed at each iteration
///@param LoopCondition Loop condition
///@param First Initial iteration value
///@param Bound Loop iteration bound
template <int NormalizedInitStep, int NormalizedExitStep, int Increment>
void inline normalized_partial_static_unroll(auto &LoopStep,
                                             auto &&LoopCondition, int First,
                                             int Bound) {
  if constexpr (NormalizedExitStep - NormalizedInitStep <= 1) {
    // Actual call step
    constexpr int LocalInc = NormalizedInitStep * Increment;
    int CallIdx = First + LocalInc;
    if (LoopCondition(CallIdx, Bound)) {
      LoopStep(CallIdx);
    }
  } else {
    // Subdivide call sequence
    constexpr int MidPoint = (NormalizedInitStep + NormalizedExitStep) / 2;
    normalized_partial_static_unroll<NormalizedInitStep, MidPoint, Increment>(
        LoopStep, LoopCondition, First, Bound);

    normalized_partial_static_unroll<MidPoint, NormalizedExitStep, Increment>(
        LoopStep, LoopCondition, First, Bound);
  }
}

///
///@brief Implementation for static_full_unrolling
///
/// Recursively build the loop by splitting the iteration range into subranges,
/// until the subrange length is one, at which step the corresponding iteration is
/// performed.
///
/// The range is normalized such that the full iteration domain corresponds to
/// range [0, total number of iterations)
///
///@tparam Minimum of the normalized subrange
///@tparam Smallest out-of-normalized-range iteration number
///@tparam Offset initial iteration value
///@tparam Increment How much to add to the iteration variable between each iteration 
///@param LoopStep Iteration computation
template <int NormalizedInitStep, int NormalizedExitStep, int Offset,
          int Increment>
void inline normalized_static_full_unrolling(auto &LoopStep) {
  if constexpr (NormalizedExitStep - NormalizedInitStep <= 1) {
    // Actual call step
    constexpr int LocalInc = NormalizedInitStep * Increment;
    int CallIdx = Offset + LocalInc;
    LoopStep(CallIdx);
  } else {
    // Subdivide call sequence
    constexpr int Midpoint = (NormalizedInitStep + NormalizedExitStep) / 2;
    normalized_static_full_unrolling<NormalizedInitStep, Midpoint, Offset,
                                         Increment>(LoopStep);

    normalized_static_full_unrolling<Midpoint, NormalizedExitStep,
                                         Offset, Increment>(LoopStep);
  }
}
} // namespace detail

///
///@brief Build a partially unrolled loop
///
///@tparam unroll_factor How many iterations are explicited in loop body
///@tparam Increment by how many is incremented the loop variable between
///        each iteration
///
///@param LoopStep Elementary computation performed at each step
///@param LoopCondition Boundary check computation
///@param StartIdx Initial iteration loop variable value
///@param Bound Bound for the loop
///
/// The function is equivalent to the following for loop :
/// \code {.cpp}
/// for (int I = StartIdX ; LoopCondition(I, Bound) ; I += Increment) {
///   LoopStep(I);
/// }
/// \endcode
///
/// Except that there are unroll_factor less iterations and the loop body
/// contains unroll_factor successive call to LoopStep (with I incremented
/// accordingly).
template <int unroll_factor = 1, int Increment = 1>
inline void partial_static_unroll(auto &LoopStep, auto &LoopCondition,
                                  int StartIdx, int Bound) {

  static_assert(unroll_factor >= 1, "Static unrolling requires a strictly "
                                   "positive unfold factor. For full "
                                   "static unrolling use "
                                   "full_static_unroll");
  static_assert(Increment != 0, "static_unroll increment cannot be zero");
  constexpr int newIncrement = unroll_factor * Increment;
  for (int i = StartIdx; LoopCondition(i, Bound); i += newIncrement) {
    detail::normalized_partial_static_unroll<0, unroll_factor, Increment>(
        LoopStep, LoopCondition, i, Bound);
  }
}

/// @brief Build the unrolled equivalent of a for loop
///
/// @tparam StartIdx Initial value of the iteration index
/// @tparam NbSteps The number of iterations to perform
/// @tparam Increment The value by which the iteration index is incremented
///         between iterations.
///
/// @param Functor Lambda containing the elementary computation step.
///
/// The generated code is equivalent to
/// \code {.cpp}
/// int IterationVal = StartIdx;
/// for (int Iteration = 0 ; Iteration < NbSteps ; Iteration++) {
///   LoopStep(IterationVal);
///   IterationVal += Increment;
/// }
/// \endcode
///
/// But no loop construct is used, all the calls to LoopStep are explicitely
/// performed.
template <int StartIdx, int NbSteps, int Increment = 1>
inline void static_full_unrolling(auto &Functor) {
  static_assert(NbSteps > 0, "Trying to unroll an iteration of less than 1 step");
  detail::normalized_static_full_unrolling<0, NbSteps, StartIdx, Increment>(
      Functor);
}
} // namespace ext::xilinx
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
