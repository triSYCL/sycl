
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
__attribute__((overloadable)) __clc_uint8_t
test___spirv_ocl_u_hadd(__clc_uint8_t args_0, __clc_uint8_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec2_uint8_t
test___spirv_ocl_u_hadd(__clc_vec2_uint8_t args_0, __clc_vec2_uint8_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec3_uint8_t
test___spirv_ocl_u_hadd(__clc_vec3_uint8_t args_0, __clc_vec3_uint8_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec4_uint8_t
test___spirv_ocl_u_hadd(__clc_vec4_uint8_t args_0, __clc_vec4_uint8_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec8_uint8_t
test___spirv_ocl_u_hadd(__clc_vec8_uint8_t args_0, __clc_vec8_uint8_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec16_uint8_t
test___spirv_ocl_u_hadd(__clc_vec16_uint8_t args_0,
                        __clc_vec16_uint8_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_uint16_t
test___spirv_ocl_u_hadd(__clc_uint16_t args_0, __clc_uint16_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec2_uint16_t
test___spirv_ocl_u_hadd(__clc_vec2_uint16_t args_0,
                        __clc_vec2_uint16_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec3_uint16_t
test___spirv_ocl_u_hadd(__clc_vec3_uint16_t args_0,
                        __clc_vec3_uint16_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec4_uint16_t
test___spirv_ocl_u_hadd(__clc_vec4_uint16_t args_0,
                        __clc_vec4_uint16_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec8_uint16_t
test___spirv_ocl_u_hadd(__clc_vec8_uint16_t args_0,
                        __clc_vec8_uint16_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec16_uint16_t
test___spirv_ocl_u_hadd(__clc_vec16_uint16_t args_0,
                        __clc_vec16_uint16_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_uint32_t
test___spirv_ocl_u_hadd(__clc_uint32_t args_0, __clc_uint32_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec2_uint32_t
test___spirv_ocl_u_hadd(__clc_vec2_uint32_t args_0,
                        __clc_vec2_uint32_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec3_uint32_t
test___spirv_ocl_u_hadd(__clc_vec3_uint32_t args_0,
                        __clc_vec3_uint32_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec4_uint32_t
test___spirv_ocl_u_hadd(__clc_vec4_uint32_t args_0,
                        __clc_vec4_uint32_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec8_uint32_t
test___spirv_ocl_u_hadd(__clc_vec8_uint32_t args_0,
                        __clc_vec8_uint32_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec16_uint32_t
test___spirv_ocl_u_hadd(__clc_vec16_uint32_t args_0,
                        __clc_vec16_uint32_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_uint64_t
test___spirv_ocl_u_hadd(__clc_uint64_t args_0, __clc_uint64_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec2_uint64_t
test___spirv_ocl_u_hadd(__clc_vec2_uint64_t args_0,
                        __clc_vec2_uint64_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec3_uint64_t
test___spirv_ocl_u_hadd(__clc_vec3_uint64_t args_0,
                        __clc_vec3_uint64_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec4_uint64_t
test___spirv_ocl_u_hadd(__clc_vec4_uint64_t args_0,
                        __clc_vec4_uint64_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec8_uint64_t
test___spirv_ocl_u_hadd(__clc_vec8_uint64_t args_0,
                        __clc_vec8_uint64_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}

__attribute__((overloadable)) __clc_vec16_uint64_t
test___spirv_ocl_u_hadd(__clc_vec16_uint64_t args_0,
                        __clc_vec16_uint64_t args_1) {
  return __spirv_ocl_u_hadd(args_0, args_1);
}
