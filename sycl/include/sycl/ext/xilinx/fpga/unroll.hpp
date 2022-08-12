//==- unroll.hpp --- SYCL Xilinx native loop unrolling       -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_UNROLL_HPP
#define SYCL_XILINX_FPGA_UNROLL_HPP

#include "sycl/detail/defines.hpp"

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {
namespace ext::xilinx {
namespace detail {
template <unsigned int unroll_factor> struct ConstrainedUnrolling {
  static_assert(
      unroll_factor > 1,
      "Constrained unrolling factor should be strictly greater than one");
  static constexpr unsigned int UnrollingFactor = unroll_factor;
  static constexpr bool FullUnroll = false;
};
} // namespace detail

template <unsigned int unroll_factor>
struct checked_fixed_unrolling
    : public detail::ConstrainedUnrolling<unroll_factor> {
  static constexpr bool checked = true;
};

template <unsigned int unroll_factor>
struct unchecked_fixed_unrolling
    : public detail::ConstrainedUnrolling<unroll_factor> {
  static constexpr bool checked = false;
};

struct full_unrolling {
  static constexpr bool FullUnroll = true;
  static constexpr unsigned int UnrollingFactor = 0;
  static constexpr bool checked = false;
};

struct no_unrolling {
  static constexpr bool FullUnroll = false;
  static constexpr unsigned int UnrollingFactor = 1;
  static constexpr bool checked = false;
};

/** Xilinx loop unrolling.

  unroll a loop

 \tparam UnrollType determines the type of unrolling to perform. Can be
 (Un)checked_fixed_unrolling<>, full_unrolling or no_unrolling

 \tparam T type of the functor to execute
*/
template <typename UnrollType = full_unrolling, typename T>
__SYCL_ALWAYS_INLINE void unroll(T &&functor) {
  __SYCL_DEVICE_ANNOTATE("xilinx_unroll", UnrollType::UnrollingFactor,
                        UnrollType::checked)
  int annotationAnchor;
  (void)annotationAnchor;
  std::forward<T>(functor)();
}

} // namespace ext::xilinx
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
