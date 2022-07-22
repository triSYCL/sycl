//==- kernel_properties.hpp --- SYCL Xilinx kernel properties       -------==//
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

#include "sycl/ext/xilinx/literals/cstr.hpp"
#include "sycl/detail/pi.h"
#include <cstddef>
#include <iostream>
#include <regex>
#include <tuple>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl::ext::xilinx {

namespace detail {
template <typename KernelType, typename Functor, typename propType,
          auto... propPayload>
struct KernelDecorator;

template <typename KernelType, typename Ret, typename Functor, typename... Args,
          typename CharT, CharT... CharPack, auto... propPayload>
struct KernelDecorator<KernelType, Ret (Functor::*)(Args...) const,
                       cstr<CharT, CharPack...>, propPayload...> {
  const KernelType kernel;
  KernelDecorator(KernelType &kernel) : kernel{kernel} {}
#ifdef __SYCL_SPIR_DEVICE__
  __SYCL_DEVICE_ANNOTATE("xilinx_kernel_property",
                         StrPacker<cstr<CharT, CharPack...>>{}.Str,
                         std::make_tuple(propPayload...))
#endif
  Ret operator()(Args... args) const { return kernel(args...); }
};

} // namespace detail

} // namespace sycl::ext::xilinx
} // __SYCL_INLINE_NAMESPACE(cl)

#endif // SYCL_XILINX_FPGA_KERNEL_PROPERTIES_HPP
