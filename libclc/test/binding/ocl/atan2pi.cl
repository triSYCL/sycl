
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Autogenerated by gen-libclc-test.py

// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

#include <spirv/spirv_types.h>

// CHECK-NOT: declare {{.*}} @_Z
// CHECK-NOT: call {{[^ ]*}} bitcast
__attribute__((overloadable)) __clc_fp32_t
test___spirv_ocl_atan2pi(__clc_fp32_t args_0, __clc_fp32_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec2_fp32_t
test___spirv_ocl_atan2pi(__clc_vec2_fp32_t args_0, __clc_vec2_fp32_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec3_fp32_t
test___spirv_ocl_atan2pi(__clc_vec3_fp32_t args_0, __clc_vec3_fp32_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec4_fp32_t
test___spirv_ocl_atan2pi(__clc_vec4_fp32_t args_0, __clc_vec4_fp32_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec8_fp32_t
test___spirv_ocl_atan2pi(__clc_vec8_fp32_t args_0, __clc_vec8_fp32_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec16_fp32_t
test___spirv_ocl_atan2pi(__clc_vec16_fp32_t args_0, __clc_vec16_fp32_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_fp64_t
test___spirv_ocl_atan2pi(__clc_fp64_t args_0, __clc_fp64_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_vec2_fp64_t
test___spirv_ocl_atan2pi(__clc_vec2_fp64_t args_0, __clc_vec2_fp64_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_vec3_fp64_t
test___spirv_ocl_atan2pi(__clc_vec3_fp64_t args_0, __clc_vec3_fp64_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_vec4_fp64_t
test___spirv_ocl_atan2pi(__clc_vec4_fp64_t args_0, __clc_vec4_fp64_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_vec8_fp64_t
test___spirv_ocl_atan2pi(__clc_vec8_fp64_t args_0, __clc_vec8_fp64_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_vec16_fp64_t
test___spirv_ocl_atan2pi(__clc_vec16_fp64_t args_0, __clc_vec16_fp64_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp16
__attribute__((overloadable)) __clc_fp16_t
test___spirv_ocl_atan2pi(__clc_fp16_t args_0, __clc_fp16_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp16
__attribute__((overloadable)) __clc_vec2_fp16_t
test___spirv_ocl_atan2pi(__clc_vec2_fp16_t args_0, __clc_vec2_fp16_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp16
__attribute__((overloadable)) __clc_vec3_fp16_t
test___spirv_ocl_atan2pi(__clc_vec3_fp16_t args_0, __clc_vec3_fp16_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp16
__attribute__((overloadable)) __clc_vec4_fp16_t
test___spirv_ocl_atan2pi(__clc_vec4_fp16_t args_0, __clc_vec4_fp16_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp16
__attribute__((overloadable)) __clc_vec8_fp16_t
test___spirv_ocl_atan2pi(__clc_vec8_fp16_t args_0, __clc_vec8_fp16_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
#ifdef cl_khr_fp16
__attribute__((overloadable)) __clc_vec16_fp16_t
test___spirv_ocl_atan2pi(__clc_vec16_fp16_t args_0, __clc_vec16_fp16_t args_1) {
  return __spirv_ocl_atan2pi(args_0, args_1);
}

#endif
