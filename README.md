# SYCL for Vitis 2021.1: experimental melting pot with Intel oneAPI DPC++ SYCL and triSYCL for Xilinx FPGA

Intel staging area for llvm.org contribution. Home for Intel LLVM-based projects:

This project is about assessing the viability of unifying 2 of the
current open-source implementations of the SYCL standard
[https://www.khronos.org/sycl](https://www.khronos.org/sycl) to
provide a strong modern single-source C++ solution for heterogeneous
computing based on Clang*/LLVM*.

All this is an experimental work-in-progress open-source research
project but might be interesting for anyone versed into system-wide
modern C++20 and heterogeneous computing involving FPGA, GPU, DSP,
other accelerators or just CPU from various vendors at the same time
in the same program.

## What is SYCL

[SYCL](https://www.khronos.org/sycl/) is a single-source
modern C++11/.../C++20-based DSEL (Domain Specific Embedded Language) aimed at
facilitating the programming of heterogeneous accelerators.

## triSYCL for Xilinx FPGA with Xilinx Vitis v++

Some LLVM passes and some C++ SYCL runtime from
https://github.com/triSYCL/triSYCL are merged-in with a new Clang
driver and scripts to use Xilinx Vitis
[v++](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/vitiscommandcompiler.html)
as a back-end for Xilinx FPGA using the open-source runtime and
device-driver https://github.com/Xilinx/XRT

## Intel oneAPI DPC++ SYCL compiler and runtime libraries using Clang/LLVM technology

This is a fork of the Intel SYCL upstreaming effort
([https://github.com/intel/llvm/tree/sycl](https://github.com/intel/llvm/tree/sycl))
with some alterations made to allow SYCL compilation for Xilinx FPGA. However,
the alterations made shouldn't affect previous targets supported by
the Intel tool, so in theory it should be possible to use different
accelerators from different vendors at the same time, including for
example an Intel FPGA and a Xilinx FPGA.

## SYCL Related Documentation

- Xilinx FPGA get started guide for the SYCL compiler
  [GettingStartedXilinxFPGA.md](sycl/doc/GettingStartedXilinxFPGA.md)
- The unchanged get started guide for the SYCL compiler
  [GetStartedGuide.md](sycl/doc/GetStartedGuide.md)
- Xilinx FPGA Tests Documentation - [Tests.md](sycl/doc/Tests.md)
  covers a few details about the the
  additional [xocc_tests](sycl/test/xocc_tests) directory we added to
  the [sycl/test](sycl/test) directory among some other small details.

The [Build DPC++ toolchain](sycl/doc/GetStartedGuide.md#build-dpc-toolchain) from the
Intel oneAPI DPC++ SYCL project is a good starting point to get to
grips with building the compiler and what a basic SYCL example looks
like.

It also showcases the requirements to get the project and examples
running with the Intel OpenCL runtime or other back-ends.

This fork of the project can be compiled the same way
and used in conjunction with the normal compiler commands as demonstrated.
However, the software requirements for Xilinx FPGA compilation and the compiler
invocation are not the same and are documented elsewhere.


## License
See [LICENSE.txt](llvm/LICENSE.TXT) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

*Other names and brands may be claimed as the property of others.

oneAPI DPC++ is an open, cross-architecture language built upon the ISO C++ and Khronos
SYCL\* standards. DPC++ extends these standards with a number of extensions,
which can be found in [sycl/doc/extensions](sycl/doc/extensions) directory.

*\*Other names and brands may be claimed as the property of others.*
