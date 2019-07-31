# Experimental melting pot of Intel SYCL* up-stream candidate and triSYCL

## Introduction

This project is about assessing the viability of unifying 2 of the
current open-source implementations of the SYCL standard
[https://www.khronos.org/sycl](https://www.khronos.org/sycl) to
provide a strong modern single-source C++ solution for heterogeneous
computing based on Clang*/LLVM*.

All this is an experimental WIP open-source research project but might
be interesting for anyone versed into system-wide modern C++20 and
heterogeneous computing involving FPGA, GPU, DSP, other accelerators
or just CPU from various vendors at the same time from the same
program.

## What is SYCL

[SYCL](https://www.khronos.org/sycl/) is a single-source
modern C++11/.../C++20-based DSEL (Domain Specific Embedded Language) aimed at
facilitating the programming of heterogeneous accelerators.

## triSYCL for Xilinx FPGA with Xilinx SDx xocc

Some LLVM passes and some C++ SYCL runtime from
https://github.com/triSYCL/triSYCL are merged-in with a new Clang
driver and scripts to use Xilinx SDx
[xocc](https://www.xilinx.com/html_docs/xilinx2019_1/sdaccel_doc/wrj1504034328013.html)
as a back-end for Xilinx FPGA using the open-source runtime and
device-driver https://github.com/Xilinx/XRT

## Intel SYCL compiler and runtime libraries using LLVM technology

This is a fork of the Intel SYCL upstreaming effort
([https://github.com/intel/llvm/tree/sycl](https://github.com/intel/llvm/tree/sycl))
with some alterations made to allow SYCL compilation for Xilinx FPGA's. However,
the alterations made shouldn't affect previous targets supported by
the Intel tool.

## SYCL Related Documentation

- Basic get started guide for the SYCL compiler - [GetStartedWithSYCLCompiler.md](sycl/doc/GetStartedWithSYCLCompiler.md)
- Xilinx FPGA get started guide for the SYCL compiler - [XilinxFPGACompilation.md](sycl/doc/XilinxFPGACompilation.md)
- Xilinx ACAP get started guide for the SYCL compiler - [GettingStartedACAP.md](sycl/doc/GettingStartedACAP.md)
- Xilinx FPGA Tests Documentation - [Tests.md](sycl/doc/Tests.md)

The [GetStartedWithSYCLCompiler.md](sycl/doc/GetStartedWithSYCLCompiler.md) is
from the Intel SYCL project and is a good starting point to get to grips with
building the compiler and what a basic SYCL example looks like. It also
showcases the requirements to get the project and examples running with the
Intel OpenCL runtime. This fork of the project an be compiled the same way
and used in conjunction with the normal compiler commands as demonstrated.
However, the software requirements for Xilinx FPGA compilation and the compiler
invocation are not the same and are documented elsewhere.

The [XilinxFPGACompilation.md](sycl/doc/XilinxFPGACompilation.md) documents the
required software and the main differences when compiling an example for Xilinx
FPGA when using the branch or some variation of it.

The [Tests.md](sycl/doc/Tests.md) covers a few details about the the
additional [xocc_tests](sycl/test/xocc_tests) directory we added to
the [sycl/test](sycl/test) directory among some other small details.

## License
See [LICENSE.txt](llvm/LICENSE.TXT) for details.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

*Other names and brands may be claimed as the property of others.

## SYCL Extension Proposal Documents

See [sycl/doc/extensions](sycl/doc/extensions)
