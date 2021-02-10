//==- memory_properties.hpp --- SYCL Xilinx memory proprerties       -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_MEMORY_PROPERTIES_HPP
#define SYCL_XILINX_FPGA_MEMORY_PROPERTIES_HPP

#include "CL/sycl/detail/defines.hpp"
#include "CL/sycl/detail/property_helper.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {
namespace xilinx {
namespace property {

struct ddr_bank {
  template <unsigned A> struct instance {
    template <int B> constexpr bool operator==(const instance<B> &) const {
      return A == B;
    }
    template <int B> constexpr bool operator!=(const instance<B> &) const {
      return A != B;
    }
  };
};
} // namespace property

template <typename... Ts>
using accessor_property_list = sycl::ONEAPI::accessor_property_list<Ts...>;

template <int A> inline constexpr property::ddr_bank::instance<A> ddr_bank;

} // namespace xilinx

namespace ONEAPI {
template <>
struct is_compile_time_property<xilinx::property::ddr_bank> : std::true_type {};
} // namespace ONEAPI

namespace detail {
template <int I>
struct IsCompileTimePropertyInstance<xilinx::property::ddr_bank::instance<I>>
    : std::true_type {};
} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
