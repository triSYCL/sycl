#!/bin/bash
# Sets up Intel SYCL related environment variables for my machine.

export OPENCL_HEADERS=/usr/include/CL
export SYCL_HOME=/storage/ogozillo/intel-sycl/sycl
export ISYCL_BIN_DIR=/storage/ogozillo/intel-sycl/build/bin
export ISYCL_LIB_DIR=/storage/ogozillo/intel-sycl/build/lib
export PATH=ISYCL_BIN_DIR:$PATH
export LD_LIBRARY_PATH=$ISYCL_LIB_DIR:$LD_LIBRARY_PATH
