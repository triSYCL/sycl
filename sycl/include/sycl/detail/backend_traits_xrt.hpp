//===------- backend_traits_xrt.hpp - Backend traits for XRT -----*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the specializations of the sycl::detail::interop,
// sycl::detail::BackendInput and sycl::detail::BackendReturn specialization for
// the XRT backend.
//
// the supported conversions are:
//  sycl::device <-> xrt::device
//  sycl::kernel <-> xrt::kernel
//
// sycl::queue, sycl::context, sycl::platform and sycl::event have no XRT
// equivalents so they are not supported
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/detail/backend_traits.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>

/// There is no cyclic dependencies, but the XRT include path is not configured
/// by the CMake of the SYCL runtime (only the pi_xrt is) while compiling this
/// file.
namespace xrt {
struct device;
struct kernel;
struct xclbin;
struct bo;
}

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::detail {

template <> struct InteropFeatureSupportMap<backend::xrt> {
  static constexpr bool MakePlatform = false;
  static constexpr bool MakeDevice = true;
  static constexpr bool MakeContext = false;
  static constexpr bool MakeQueue = false;
  static constexpr bool MakeEvent = false;
  static constexpr bool MakeKernelBundle = true;
  static constexpr bool MakeKernel = true;
  static constexpr bool MakeBuffer = true;
};

template <> struct BackendInput<backend::xrt, device> {
  using type = const xrt::device&;
};

template <> struct BackendReturn<backend::xrt, device> {
  using type = const xrt::device&;
};

template <> struct BackendInput<backend::xrt, sycl::kernel> {
  using type = const xrt::kernel&;
};

template <> struct BackendReturn<backend::xrt, sycl::kernel> {
  using type = const xrt::kernel&;
};

template <>
struct BackendInput<backend::xrt,
                    sycl::kernel_bundle<sycl::bundle_state::executable>> {
  using type = const xrt::xclbin&;
};

template <>
struct BackendReturn<backend::xrt,
                     sycl::kernel_bundle<sycl::bundle_state::executable>> {
  using type = std::vector<xrt::xclbin>;
};

} // namespace detail
} // __SYCL_INLINE_NAMESPACE(cl)
