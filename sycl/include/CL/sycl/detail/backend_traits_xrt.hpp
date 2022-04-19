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
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/queue.hpp>

namespace xrt {
struct device;
struct xclbin;
struct bo;
}

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <> struct InteropFeatureSupportMap<backend::xrt> {
  static constexpr bool MakePlatform = false;
  static constexpr bool MakeDevice = true;
  static constexpr bool MakeContext = false;
  static constexpr bool MakeQueue = false;
  static constexpr bool MakeEvent = false;
  static constexpr bool MakeKernelBundle = false;
  static constexpr bool MakeKernel = true;
  static constexpr bool MakeBuffer = false;
};

template <> struct interop<backend::xrt, context> {
};

template <> struct interop<backend::xrt, device> {
  using type = xrt::device*;
};

template <> struct interop<backend::xrt, event> {
};

template <> struct interop<backend::xrt, queue> {
};

// TODO the interops for accessor is used in the already deprecated class
// interop_handler and can be removed after API cleanup.
template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::xrt,
               accessor<DataT, Dimensions, AccessMode, access::target::device,
                        access::placeholder::false_t>> {
  using type = xrt::bo*;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<
    backend::xrt,
    accessor<DataT, Dimensions, AccessMode, access::target::constant_buffer,
             access::placeholder::false_t>> {
  using type = xrt::bo*;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::xrt,
                    buffer<DataT, Dimensions, AllocatorT>> {
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::xrt,
                     buffer<DataT, Dimensions, AllocatorT>> {
};

template <> struct BackendInput<backend::xrt, context> {
};

template <> struct BackendReturn<backend::xrt, context> {
};

template <> struct BackendInput<backend::xrt, device> {
  using type = void*;
};

template <> struct BackendReturn<backend::xrt, device> {
  using type = xrt::device&;
};

template <> struct BackendInput<backend::xrt, event> {
};

template <> struct BackendReturn<backend::xrt, event> {
};

template <> struct BackendInput<backend::xrt, queue> {
};

template <> struct BackendReturn<backend::xrt, queue> {
};

#ifdef __SYCL_INTERNAL_API
template <> struct BackendInput<backend::xrt, program> {
};

template <> struct BackendReturn<backend::xrt, program> {
};
#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
