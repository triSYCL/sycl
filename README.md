# SYCL for Vitis 2022.1: experimental melting pot with Intel oneAPI DPC++ SYCL and triSYCL for AMD/Xilinx FPGA

This project is about assessing the viability of unifying 2 of the
current open-source implementations of the SYCL standard
[https://www.khronos.org/sycl](https://www.khronos.org/sycl) to
provide a strong modern single-source C++ solution for heterogeneous
computing based on Clang*/LLVM*.

All this is an experimental work-in-progress open-source research
project but might be interesting for anyone versed into system-wide
modern C++20/C++23 and heterogeneous computing involving FPGA, GPU, DSP,
other accelerators or just CPU from various vendors at the same time
in the same program.

There are mostly 2 public branches:

- https://github.com/triSYCL/sycl/tree/sycl/unified/next is where we
  are doing our latest developments and where you can get the latest
  features for the latest platforms and OS. This is where you should
  open your pull-requests;
- https://github.com/triSYCL/sycl/tree/sycl/unified/master is the more
  stable and older version.


## What is SYCL

[SYCL](https://www.khronos.org/sycl/) is a single-source
modern C++11/.../C++20-based DSEL (Domain Specific Embedded Language) aimed at
facilitating the programming of heterogeneous accelerators.

## triSYCL for AMD/Xilinx FPGA with AMD/Xilinx Vitis v++

Some LLVM passes and some C++ SYCL runtime from
https://github.com/triSYCL/triSYCL are merged-in with a new Clang
driver and scripts to use AMD/Xilinx Vitis
[v++](https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Vitis-Compiler-Command)
as a back-end for AMD/Xilinx FPGA using the open-source runtime and
device-driver https://github.com/Xilinx/XRT

## Intel oneAPI DPC++ SYCL compiler and runtime libraries using Clang/LLVM technology

This is a fork of the Intel SYCL upstreaming effort
([https://github.com/intel/llvm/tree/sycl](https://github.com/intel/llvm/tree/sycl))
with some alterations made to allow SYCL compilation for AMD/Xilinx
FPGA. However, the alterations made shouldn't affect previous targets
supported by the Intel tool, so in theory it should be possible to use
different accelerators from different vendors at the same time,
including for example an Intel FPGA and an AMD/Xilinx FPGA.

## SYCL Related Documentation

- AMD/Xilinx FPGA get started guide for the SYCL compiler
  [GettingStartedXilinxFPGA.md](sycl/doc/GettingStartedXilinxFPGA.md)
- The unchanged get started guide for the SYCL compiler
  [GetStartedGuide.md](sycl/doc/GetStartedGuide.md)

- AMD/Xilinx FPGA Tests Documentation
  - [Tests.md](sycl/doc/Tests.md) covers a few details about the the
    additional [vitis](sycl/test/vitis) directory we added
    to the [sycl/test](sycl/test) directory among some other small
    details.

The [Build DPC++ toolchain](sycl/doc/GetStartedGuide.md#build-dpc-toolchain) from the
Intel oneAPI DPC++ SYCL project is a good starting point to get to
grips with building the compiler and what a basic SYCL example looks
like.

It also showcases the requirements to get the project and examples
running with the Intel OpenCL runtime or other back-ends.

This fork of the project can be compiled the same way and used in
conjunction with the normal compiler commands as demonstrated.
However, the software requirements for AMD/Xilinx FPGA compilation and
the compiler invocation are not the same and are documented elsewhere.


## License
See [LICENSE.txt](llvm/LICENSE.TXT) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

*Other names and brands may be claimed as the property of others.

oneAPI DPC++ is an open, cross-architecture language built upon the ISO C++ and Khronos
SYCL\* standards. DPC++ extends these standards with a number of extensions,
which can be found in [sycl/doc/extensions](sycl/doc/extensions) directory.

*\*Other names and brands may be claimed as the property of others.*
