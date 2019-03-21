Test Cases
++++++++++

Commands
========

These tests should all be executable in conjunction with the Xilinx SDAccel and
Vivado toolchains alongside an installation of the XRT run-time. The
SYCL Compiler using the following command to compile using XOCC:

$SYCL_BIN_DIR/clang++ -D__SYCL_SPIR_DEVICE__ -std=c++17 -fsycl \
 -fsycl-xocc-device test-case.cpp -o test-case -lsycl -lOpenCL

Two step compile is currently a work in progress.

The tests should also function for the Intel OpenCL run-time/SDK using the
standard command from the Intel SYCL up-streaming effort:

$ISYCL_BIN_DIR/clang++ -std=c++17 -fsycl test-case.cpp -o test-case \
  -lsycl -lOpenCL

It's noteworthy that the SYCL run-time is compiled using C++20, most of the
current features are C++11 compatible outside of the Xilinx vendor components
for the moment. This is likely to change however. You can also only compile
source code with C++11 to C++17 for the time being as the standard library
wrapper stubs need to be updated for C++20.

You'll also have to swap out the device selector for the test as using an
XOCCDeviceSelector with an Intel (SPIRV) compiled executeable will cause
the program_manager to emit a run-time error when it feeds the kernel 
binary to the wrong OpenCL run-time.

Test Directories
================

All the tests in the simple_tests directory are small unit tests with the
intention of testing certain functionality works for the Xilinx toolchain.

The tests in sdaccel_ports are rough implementations of tests in the
https://github.com/Xilinx/SDAccel_Examples repository.

The issue_related directory is mostly for development purposes to hold test
cases that are known to have problems, either because of unclear specification
behavior, compiler or run-time bugs. Once the issues are fixed the test cases
should be moved into simple_tests where possible.

The utilities directory is mostly for abstractions or functionality that's
used inside the test cases but has no place in the SYCL run-time just yet.

Feel free to contribute other tests that you interesting or come alongside fixed
issues you have encountered. Also happy to take in more complex examples!
