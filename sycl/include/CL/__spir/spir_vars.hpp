//==---------- spir_vars.hpp --- SPIR variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

// TODO: This and spirv_vars will probably need refined a little when the issue:
// https://github.com/intel/llvm/issues/253 has been discussed a little more.
#ifdef __SYCL_DEVICE_ONLY__
// This is intended as a drop in replacement for SPIRV vars on SPIR devices.
// Although the __spirv_BuiltInSubgroupSize builtins at the end of the file
// currently do not have a SPIR variation, so they will break at the moment,
// they're included here so as not to break the compilation for the moment
// (although it may be more desirable than a runtime error/symbol missing
//  error)

// These are converted to SPIR builtins by the inSPIRation pass.
size_t __spir_ocl_get_global_size(unsigned int dimindx);
size_t __spir_ocl_get_local_size(unsigned int dimindx);
size_t __spir_ocl_get_global_id(unsigned int dimindx);
size_t __spir_ocl_get_local_id(unsigned int dimindx);
size_t __spir_ocl_get_global_offset(unsigned int dimindx);
size_t __spir_ocl_get_group_id(unsigned int dimindx);
size_t __spir_ocl_get_num_groups(unsigned int dimindx);

enum class SYCLBuiltinTypes {
   SYCLBuiltinGlobalSize,
   SYCLBuiltinGlobalInvocationId,
   SYCLBuiltinWorkgroupSize,
   SYCLBuiltinNumWorkgroups,
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
    case SYCLBuiltinTypes::SYCLBuiltinNumWorkgroups:
      return __spir_ocl_get_num_groups(ID);
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

namespace __spir {

DEFINE_SYCL_SPIR_CONVERTER(GlobalSize);
DEFINE_SYCL_SPIR_CONVERTER(GlobalInvocationId)
DEFINE_SYCL_SPIR_CONVERTER(WorkgroupSize)
DEFINE_SYCL_SPIR_CONVERTER(NumWorkgroups)
DEFINE_SYCL_SPIR_CONVERTER(LocalInvocationId)
DEFINE_SYCL_SPIR_CONVERTER(WorkgroupId)
DEFINE_SYCL_SPIR_CONVERTER(GlobalOffset)

}

#undef DEFINE_SYCL_SPIR_CONVERTER

// SPIRV intrinsics/builtins that we currently don't map for yet (unsure if
// they have equivalent SPIR or HLS mappings).
// TODO: Look into these to see if we have equivalents
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupSize;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupMaxSize;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInNumSubgroups;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInNumEnqueuedSubgroups;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupId;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

#define DEFINE_INIT_SIZES(POSTFIX)                                             \
                                                                               \
  template <int Dim, class DstT> struct InitSizesST##POSTFIX;                  \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<1, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<0>()}; }                     \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<2, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<0>(), get##POSTFIX<1>()}; }  \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<3, DstT> {                 \
    static DstT initSize() {                                                   \
      return {get##POSTFIX<0>(), get##POSTFIX<1>(), get##POSTFIX<2>()};        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <int Dims, class DstT> static DstT init##POSTFIX() {                \
    return InitSizesST##POSTFIX<Dims, DstT>::initSize();                       \
  }

namespace __spir {

DEFINE_INIT_SIZES(GlobalSize);
DEFINE_INIT_SIZES(GlobalInvocationId)
DEFINE_INIT_SIZES(WorkgroupSize)
DEFINE_INIT_SIZES(NumWorkgroups)
DEFINE_INIT_SIZES(LocalInvocationId)
DEFINE_INIT_SIZES(WorkgroupId)
DEFINE_INIT_SIZES(GlobalOffset)

} // namespace __spir

#endif // __SYCL_DEVICE_ONLY__
