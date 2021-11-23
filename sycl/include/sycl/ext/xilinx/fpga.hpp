//==- fpga.hpp --- SYCL Xilinx FPGA vendor extensions header         -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The main include header for Xilinx FPGA extensions
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_HPP
#define SYCL_XILINX_FPGA_HPP

#include "sycl/ext/xilinx/fpga/kernel_param.hpp"
#include "sycl/ext/xilinx/fpga/dataflow.hpp"
#include "sycl/ext/xilinx/fpga/kernel_properties.hpp"
#include "sycl/ext/xilinx/fpga/memory_properties.hpp"
#include "sycl/ext/xilinx/fpga/partition_array.hpp"
#include "sycl/ext/xilinx/fpga/pipeline.hpp"
#include "sycl/ext/xilinx/fpga/static_unroll.hpp"
#include "sycl/ext/xilinx/fpga/unroll.hpp"

#endif // SYCL_XILINX_FPGA_HPP
