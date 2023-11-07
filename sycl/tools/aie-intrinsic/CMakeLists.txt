# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# AIE intrinsics are only known to the chess compiler and not to our compiler.
# Thus, to be able to use them, we build this library with the Synopsys Chess compiler
#  containing all the intrinsics we use and link it with our device code in sycl-chess.

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.cpp ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h)

if ("aie" IN_LIST SYCL_ENABLE_PLUGINS)
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/aie-intrinsic.o
                   COMMAND xchesscc -p me -C Release_LLVM -d -f -c -o ${CMAKE_BINARY_DIR}/lib/aie-intrinsic.o ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.cpp
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.cpp ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h
                   COMMENT "Built AIE intrinsics library")
  add_custom_target(aie-intrinsic ALL DEPENDS ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h ${CMAKE_BINARY_DIR}/lib/aie-intrinsic.o)
else()
  add_custom_target(aie-intrinsic ALL DEPENDS ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h)
endif()

