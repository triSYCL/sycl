//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__


// TODO: This should perhaps be moved to a spir_vars.hpp file and then
// optionally include spir_vars or spirv_vars based on the device being compiled
// for.
#ifdef __SYCL_SPIR_DEVICE__
size_t __spir_ocl_get_global_size(uint dimindx);
size_t __spir_ocl_get_local_size(uint dimindx);
size_t __spir_ocl_get_global_id(uint dimindx);
size_t __spir_ocl_get_local_id(uint dimindx);
size_t __spir_ocl_get_global_offset(uint dimindx);
size_t __spir_ocl_get_group_id(uint dimindx);
#endif

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
extern "C" const __constant size_t_vec __spirv_BuiltInGlobalSize;
extern "C" const __constant size_t_vec __spirv_BuiltInGlobalInvocationId;
extern "C" const __constant size_t_vec __spirv_BuiltInWorkgroupSize;
extern "C" const __constant size_t_vec __spirv_BuiltInLocalInvocationId;
extern "C" const __constant size_t_vec __spirv_BuiltInWorkgroupId;
extern "C" const __constant size_t_vec __spirv_BuiltInGlobalOffset;

// Depending on how much you like macros this may make you smile or vomit. The
// SYCL handlers call some varation of get##POSTFIX() defined by
// DEFINE_SYCL_GET_BUILTIN which then calls MapTo defined by
// DEFINE_MAP_TO_BUILTIN_CONVERTER which forwards to either SPIR or SPIRV
// builtins based on the device it's compiled for. The builtins are wrapped in
// prefixed functions defined by DEFINE_SPIRV_CONVERTER and
// DEFINE_SPIR_CONVERTER respectively (mostly to get around the fact that SPIRV
// calls are a struct/vec type and OpenCL SPIR calls are functions)
//
// Aside from being able to remove the SPIR calls from being directly added to
// the SYCL handlers this should make it easier to seperate these into
// spir_vars.hpp/spirv_vars.hpp/other_builtin_vars.hpp
//
// Note: If this becomes a problem when merging it's probably quite possible to
// have a seperation of concerns let the not have to
//
enum SYCLBuiltinTypes {
   SYCLBuiltinGlobalSize,
   SYCLBuiltinGlobalInvocationId,
   SYCLBuiltinWorkgroupSize,
   SYCLBuiltinLocalInvocationId,
   SYCLBuiltinWorkgroupId,
   SYCLBuiltinGlobalOffset
};

#define DEFINE_SPIRV_CONVERTER(NAME, POSTFIX)                                 \
  template <int ID> static size_t __spirv##NAME();                            \
  template <> size_t __spirv##NAME<0>() { return __spirv_BuiltIn##POSTFIX.x; }\
  template <> size_t __spirv##NAME<1>() { return __spirv_BuiltIn##POSTFIX.y; }\
  template <> size_t __spirv##NAME<2>() { return __spirv_BuiltIn##POSTFIX.z; }

#define DEFINE_SPIR_CONVERTER(NAME, POSTFIX)                            \
  template <int ID> static size_t __spir##NAME() {                      \
    return __spir_ocl_##POSTFIX(ID);                                    \
  }

#define DEFINE_MAP_TO_BUILTIN_CONVERTER(PREFIX)               \
template <int ID>                                             \
size_t MapTo(SYCLBuiltinTypes builtin) {                      \
  switch (builtin) {                                          \
    case SYCLBuiltinGlobalSize:                               \
        return __##PREFIX##SYCLBuiltinGlobalSize<ID>();       \
    break;                                                    \
    case SYCLBuiltinGlobalInvocationId:                       \
      return __##PREFIX##SYCLBuiltinGlobalInvocationId<ID>(); \
    break;                                                    \
    case SYCLBuiltinWorkgroupSize:                            \
      return __##PREFIX##SYCLBuiltinWorkgroupSize<ID>();      \
    break;                                                    \
    case SYCLBuiltinLocalInvocationId:                        \
      return __##PREFIX##SYCLBuiltinLocalInvocationId<ID>();  \
    break;                                                    \
    case SYCLBuiltinWorkgroupId:                              \
      return __##PREFIX##SYCLBuiltinWorkgroupId<ID>();        \
    break;                                                    \
    case SYCLBuiltinGlobalOffset:                             \
      return __##PREFIX##SYCLBuiltinGlobalOffset<ID>();       \
    break;                                                    \
  }                                                           \
}

#define DEFINE_SYCL_GET_BUILTIN(POSTFIX)                  \
  template <int ID> static size_t get##POSTFIX() {        \
    return MapTo<ID>(SYCLBuiltin##POSTFIX);               \
  }

#ifdef __SYCL_SPIR_DEVICE__
DEFINE_SPIR_CONVERTER(SYCLBuiltinGlobalSize, get_global_size);
DEFINE_SPIR_CONVERTER(SYCLBuiltinGlobalInvocationId, get_global_id)
DEFINE_SPIR_CONVERTER(SYCLBuiltinWorkgroupSize, get_local_size)
DEFINE_SPIR_CONVERTER(SYCLBuiltinLocalInvocationId, get_local_id)
DEFINE_SPIR_CONVERTER(SYCLBuiltinWorkgroupId, get_group_id)
DEFINE_SPIR_CONVERTER(SYCLBuiltinGlobalOffset, get_global_offset)

DEFINE_MAP_TO_BUILTIN_CONVERTER(spir)
#else
DEFINE_SPIRV_CONVERTER(SYCLBuiltinGlobalSize, GlobalSize);
DEFINE_SPIRV_CONVERTER(SYCLBuiltinGlobalInvocationId, GlobalInvocationId)
DEFINE_SPIRV_CONVERTER(SYCLBuiltinWorkgroupSize, WorkgroupSize)
DEFINE_SPIRV_CONVERTER(SYCLBuiltinLocalInvocationId, LocalInvocationId)
DEFINE_SPIRV_CONVERTER(SYCLBuiltinWorkgroupId, WorkgroupId)
DEFINE_SPIRV_CONVERTER(SYCLBuiltinGlobalOffset, GlobalOffset)

DEFINE_MAP_TO_BUILTIN_CONVERTER(spirv)
#endif

DEFINE_SYCL_GET_BUILTIN(GlobalSize);
DEFINE_SYCL_GET_BUILTIN(GlobalInvocationId)
DEFINE_SYCL_GET_BUILTIN(WorkgroupSize)
DEFINE_SYCL_GET_BUILTIN(LocalInvocationId)
DEFINE_SYCL_GET_BUILTIN(WorkgroupId)
DEFINE_SYCL_GET_BUILTIN(GlobalOffset)

#undef DEFINE_SPIR_CONVERTER
#undef DEFINE_SPIRV_CONVERTER
#undef DEFINE_MAP_TO_BUILTIN_CONVERTER
#undef DEFINE_SYCL_GET_BUILTIN


extern "C" const __constant uint32_t __spirv_BuiltInSubgroupSize;
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupMaxSize;
extern "C" const __constant uint32_t __spirv_BuiltInNumSubgroups;
extern "C" const __constant uint32_t __spirv_BuiltInNumEnqueuedSubgroups;
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupId;
extern "C" const __constant uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

#endif // __SYCL_DEVICE_ONLY__
