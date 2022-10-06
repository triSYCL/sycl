//==- memory_properties.hpp --- SYCL Xilinx memory proprerties       -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_MEMORY_PROPERTIES_HPP
#define SYCL_XILINX_FPGA_MEMORY_PROPERTIES_HPP

#include "sycl/detail/defines.hpp"
#include "sycl/properties/accessor_properties.hpp"

__SYCL_INLINE_NAMESPACE(cl) {

// TODO(lforg37): [Technical Debt]: Merge properties for hbm_bank and ddr_bank
//                                  with a template type for the type of memory
//                                  bank.
//

namespace sycl {
namespace ext {
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

struct hbm_bank {
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
using accessor_property_list = sycl::ext::oneapi::accessor_property_list<Ts...>;

template <int A> inline constexpr property::ddr_bank::instance<A> ddr_bank;


template <int A> inline constexpr property::hbm_bank::instance<A> hbm_bank;

} // namespace xilinx

namespace oneapi {
template <>
struct is_compile_time_property<xilinx::property::ddr_bank> : std::true_type {};

template <>
struct is_compile_time_property<xilinx::property::hbm_bank> : std::true_type {};
} // namespace oneapi
} // namespace ext

namespace detail {
template <int I>
struct IsCompileTimePropertyInstance<ext::xilinx::property::ddr_bank::instance<I>>
    : std::true_type {};

template <int I>
struct IsCompileTimePropertyInstance<ext::xilinx::property::hbm_bank::instance<I>>
    : std::true_type {};
} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
