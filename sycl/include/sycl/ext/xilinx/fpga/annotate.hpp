//==- dataflow.hpp --- SYCL Xilinx extension -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This contains utility to apply multiple annotations on a loop
///
//===----------------------------------------------------------------------===//

#include <utility>
#include "sycl/detail/defines.hpp"

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext::xilinx {

template <typename... params, typename T> void annot(T func) {
  (params([] {}), ...);
  func();
}
}
}
}
