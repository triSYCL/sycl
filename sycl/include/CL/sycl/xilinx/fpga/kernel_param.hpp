//==- kernel_param.hpp --- SYCL Xilinx extension ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes decorating function to allow specifying arguments to
/// v++ from SYCL program source.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_KERNEL_PARAM
#define SYCL_XILINX_FPGA_KERNEL_PARAM

#include <cstdint>
#include <type_traits>
#include <utility>

#include "CL/sycl/detail/defines.hpp"
#include "CL/sycl/detail/property_helper.hpp"

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {
namespace xilinx {

template <typename... Params> auto kernel_param(auto kernel, Params...) {
  using namespace detail::literals;
  using kernelType = std::remove_cvref_t<decltype(kernel)>;
  using concat_type = detail::concat_t<detail::cstr<char, ' '>, Params...>;
  using concatenated_strpack = detail::StrPacker<concat_type>;
  return detail::KernelDecorator<kernelType, decltype(&kernelType::operator()),
                                 decltype("kernel_param"_cstr),
                                 concatenated_strpack::Str>{kernel};
}

template <typename CharT, CharT... charpack> struct KernelArgumentDecorator {
  auto operator()(auto kernel) const {
    using namespace detail::literals;
    using kernelType = std::remove_cvref_t<decltype(kernel)>;
    using argument = detail::StrPacker<detail::cstr<CharT, charpack...>>;
    return detail::KernelDecorator<
        kernelType, decltype(&kernelType::operator()),
        decltype("kernel_param"_cstr), argument::Str>{kernel};
  }
};

inline namespace literals {
/// We are using a compiler extention here and we don't want any warning about
/// it so we suppress that warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-string-literal-operator-template"

template <typename CharT, CharT... charpack>
__SYCL_ALWAYS_INLINE constexpr KernelArgumentDecorator<CharT, charpack...>
operator"" _vitis_option() noexcept {
  return {};
}

#pragma clang diagnostic pop
} // namespace literals
} // namespace xilinx
} // namespace sycl

} // __SYCL_INLINE_NAMESPACE(cl)

#endif
