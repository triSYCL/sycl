//==--------- xrt.hpp - SYCL XRT backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend.hpp>
#include <sycl/detail/pi.h>
#include <sycl/program.hpp>

#include <xrt.h>
#include <xrt/xrt_kernel.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/// The backend is responsible for selecting the correct type and reference
/// kind based on the semantic of the underlying pi call.
template <typename To>
typename std::enable_if_t<!std::is_reference_v<To>, To>
from_native_handle(pi_native_handle handle) {
  return { handle };
}

template <typename To>
typename std::enable_if_t<std::is_reference_v<To>, To>
from_native_handle(pi_native_handle handle) {
  /// A safe and convenient default is const& with the backend making a copy of
  /// what it receives. If the backend says it should be an T&& meaning the
  /// backend transfers ownership to the user, we need to let it happen so we
  /// std::forward
  return std::forward<To>(
      *reinterpret_cast<std::remove_reference_t<To> *>(handle));
}

template <typename From>
typename std::enable_if<!std::is_reference_v<From>, pi_native_handle>::type
to_native_handle(From &&from) {
  static_assert(sizeof(From) <= sizeof(pi_native_handle), "doesn't fit in pi_native_handle");
  // It can be either a static_cast or a reinterpret_cast depending on From and
  // it is intended this way
  return (pi_native_handle)from;
}

template <typename From>
typename std::enable_if<std::is_reference_v<From>, pi_native_handle>::type
to_native_handle(From &&from) {
  /// A safe and convenient default is const& with the backend returning without
  /// copies. a copy will occur on the user's side if needed. but we "handle"
  /// r-value references assuming the backend know it needs to move from it.
  return reinterpret_cast<pi_native_handle>(std::addressof(from));
}

}  // namespace detail

template <>
inline device make_device<backend::xrt>(
    const backend_input_t<backend::xrt, device> &BackendObject) {
  return detail::make_device(
      detail::to_native_handle<backend_input_t<backend::xrt, device>>(
          BackendObject),
      backend::xrt);
}

template <>
inline auto get_native<backend::xrt>(const device &Obj)
    -> backend_return_t<backend::xrt, device> {
  if (Obj.get_backend() != backend::xrt)
    throw runtime_error(errc::backend_mismatch, "Backends mismatch",
                        PI_INVALID_OPERATION);

  return detail::from_native_handle<
      backend_return_t<backend::xrt, device>>(
      Obj.getNative());
}

template <>
inline kernel make_kernel<backend::xrt>(
    const backend_input_t<backend::xrt, kernel> &BackendObject,
    const context &ctx) {
  return detail::make_kernel(
      detail::to_native_handle<backend_input_t<backend::xrt, kernel>>(
          BackendObject),
      ctx, backend::xrt);
}

template <>
inline auto get_native<backend::xrt>(const kernel &Obj)
    -> backend_return_t<backend::xrt, kernel> {
  if (Obj.get_backend() != backend::xrt)
    throw runtime_error(errc::backend_mismatch, "Backends mismatch",
                        PI_INVALID_OPERATION);

  return detail::from_native_handle<
      backend_return_t<backend::xrt, kernel>>(
      Obj.getNative());
}

template <>
inline kernel_bundle<bundle_state::executable>
make_kernel_bundle<backend::xrt, bundle_state::executable>(
    const backend_input_t<backend::xrt, kernel_bundle<bundle_state::executable>>
        &BackendObject,
    const context &ctx) {
  return detail::createSyclObjFromImpl<kernel_bundle<bundle_state::executable>>(
      detail::make_kernel_bundle(
          detail::to_native_handle<backend_input_t<
              backend::xrt, kernel_bundle<bundle_state::executable>>>(
              BackendObject),
          ctx, bundle_state::executable, backend::xrt));
}

template <>
inline auto get_native<backend::xrt>(const kernel_bundle<bundle_state::executable> &Obj)
    -> backend_return_t<backend::xrt, kernel_bundle<bundle_state::executable>> {
  if (Obj.get_backend() != backend::xrt)
    throw runtime_error(errc::backend_mismatch, "Backends mismatch",
                        PI_INVALID_OPERATION);

  return Obj.getNative<backend::xrt>([](pi_native_handle handle) {
    return detail::from_native_handle<typename backend_return_t<
        backend::xrt, kernel_bundle<bundle_state::executable>>::reference>(handle);
  });
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
