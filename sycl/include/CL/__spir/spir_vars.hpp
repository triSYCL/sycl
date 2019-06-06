//==---------- spir_vars.hpp --- SPIR variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__
// This is intended as a drop in replacement for SPIRV vars on SPIR devices.
// Although the __spirv_BuiltInSubgroupSize builtins at the end of the file
// currently do not have a SPIR variation, so they will break at the moment,
// they're included here so as not to break the compilation for the moment
// (although it may be more desirable than a runtime error/symbol missing
//  error)

// These are converted to SPIR builtins by the inSPIRation pass.
size_t __spir_ocl_get_global_size(uint dimindx);
size_t __spir_ocl_get_local_size(uint dimindx);
size_t __spir_ocl_get_global_id(uint dimindx);
size_t __spir_ocl_get_local_id(uint dimindx);
size_t __spir_ocl_get_global_offset(uint dimindx);
size_t __spir_ocl_get_group_id(uint dimindx);

enum class SYCLBuiltinTypes {
   SYCLBuiltinGlobalSize,
   SYCLBuiltinGlobalInvocationId,
   SYCLBuiltinWorkgroupSize,
   SYCLBuiltinLocalInvocationId,
   SYCLBuiltinWorkgroupId,
   SYCLBuiltinGlobalOffset
};

// Just infrastructure to use the same calls as the SPIRV builtins inside
// the handler and then redirect to the relevant SPIR builtins instead.
//
// Note: There is an arguably more powerful but harder to maintain
// implementation of this in a prior commit that could allow better
// redirection/offloading to arbitrary builtins, if the need arises.
constexpr size_t MapTo(SYCLBuiltinTypes builtin, int ID) {
  switch (builtin) {
    case SYCLBuiltinTypes::SYCLBuiltinGlobalSize:
        return __spir_ocl_get_global_size(ID);
    break;
    case SYCLBuiltinTypes::SYCLBuiltinGlobalInvocationId:
      return __spir_ocl_get_global_id(ID);
    break;
    case SYCLBuiltinTypes::SYCLBuiltinWorkgroupSize:
      return __spir_ocl_get_local_size(ID);
    break;
    case SYCLBuiltinTypes::SYCLBuiltinLocalInvocationId:
      return __spir_ocl_get_local_id(ID);
    break;
    case SYCLBuiltinTypes::SYCLBuiltinWorkgroupId:
      return __spir_ocl_get_group_id(ID);
    break;
    case SYCLBuiltinTypes::SYCLBuiltinGlobalOffset:
      return __spir_ocl_get_global_offset(ID);
    break;
  }
}

#define DEFINE_SYCL_SPIR_CONVERTER(POSTFIX)                                 \
  template <int ID> static size_t get##POSTFIX() {                          \
    return MapTo(SYCLBuiltinTypes::SYCLBuiltin##POSTFIX, ID);               \
  }

DEFINE_SYCL_SPIR_CONVERTER(GlobalSize);
DEFINE_SYCL_SPIR_CONVERTER(GlobalInvocationId)
DEFINE_SYCL_SPIR_CONVERTER(WorkgroupSize)
DEFINE_SYCL_SPIR_CONVERTER(LocalInvocationId)
DEFINE_SYCL_SPIR_CONVERTER(WorkgroupId)
DEFINE_SYCL_SPIR_CONVERTER(GlobalOffset)

#undef DEFINE_SYCL_SPIR_CONVERTER

// SPIRV intrinsics/builtins that we currently don't map for yet (unsure if
// they have equivalent SPIR or HLS mappings).
// TODO: Look into these to see if we have equivalents
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupSize;
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupMaxSize;
extern "C" const __constant uint32_t __spirv_BuiltInNumSubgroups;
extern "C" const __constant uint32_t __spirv_BuiltInNumEnqueuedSubgroups;
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupId;
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

#endif // __SYCL_DEVICE_ONLY__
