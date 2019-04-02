# Experimental melting pot of Intel SYCL* up-stream candidate and triSYCL

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

See [GetStartedWithSYCLCompiler.md](sycl/doc/GetStartedWithSYCLCompiler.md)


## triSYCL for Xilinx FPGA with Xilinx SDx xocc

Some LLVM passes and some C++ SYCL runtime from
https://github.com/triSYCL/triSYCL are merged-in with a new Clang
driver and scripts to use Xilinx SDx
[xocc](https://www.xilinx.com/html_docs/xilinx2018_3/sdaccel_doc/wrj1504034328013.html#wrj1504034328013)
as a back-end for Xilinx FPGA using the open-source runtime and
device-driver https://github.com/Xilinx/XRT


## Intel SYCL compiler and runtimes libraries using LLVM technology

This comes mainly unchanged from
https://github.com/intel/llvm/tree/sycl (see the **sycl** branch), so
it should also work for the previous targets supported by this tool.


### License

See
[LICENSE.txt](intel/llvm/LICENSE.TXT)
for details.


### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.


*Other names and brands may be claimed as the property of others.
