# Xilinx FPGA Compilation

This document aims to cover the key differences of compiling SYCL for Xilinx
FPGAs. Things like building the compiler and library remain the same but other
things like the compiler invocation for Xilinx FPGA compilation is a little
different. As a general rule of thumb we're trying to keep things as close as we
can to the Intel implementation, but in some areas were still working on that.

One of the significant differences of compilation for Xilinx FPGAs over the
ordinary compiler directive is that Xilinx devices require offline compilation
of SYCL kernels to binary before being wrapped into the end fat binary. The
offline compilation of these kernels is done by Xilinx's `xocc` compiler rather
than the SYCL device compiler itself in this case. The device compilers job is
to compile SYCL kernels to a format edible by `xocc`, then take the output of
`xocc` and wrap it into the fat binary as normal.

Xilinx's `xocc` compiler unfortunately doesn't take SPIR-V which is what raises
some problems (among other idiosyncrasies) as the current SYCL implementation
revolves around SPIR-V. It's main method of consumption is SPIR-df a slightly
modified version of LLVM-IR. So a lot of our modifications revolve around being
the middle man between `xocc` and the SYCL device compiler and runtime for the
moment, they are not the simple whims of the insane! Hopefully..

## Software requirements

Installing Xilinx FPGA compatible software stack:
  1. OpenCL headers: On Ubuntu/Debian this can be done by installing the
  opencl-c-headers package, e.g. `apt install opencl-c-headers`.
  Alternatively the headers can be download from
  [github.com/KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
  2. Xilinx runtime (XRT) for FPGAs: Download, build and install [XRT](https://github.com/Xilinx/XRT), this contains the OpenCL runtime.
  3. Xilinx SDx (2018.3+): Download and Install [SDx](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/sdx-development-environments.html) which contains the `xocc` compiler.

## Platforms

It's of note that the SDx 2018.3 install comes with several platforms that do
not work with the SYCL compiler, i.e. the ZYNC family of boards. Instead, you'll
have to use one of the newer boards, like the Alveo U250 (xilinx_u250_xdma_201830_1).
This requires some additional installation steps for the moment as it doesn't come
packaged with the SDx download for the moment.

How to:
  1) Download the Deployment and Development Shell for your OS from the [Alveo U250 getting started page](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted).
    Note: you can also install the release version of the XRT runtime from here if you would rather use this than build from source.
  2) a) Install XRT if you haven't already: ``sudo apt install xrt_201830.2.1.1695_<OS>-xrt.deb``
    b) Install the Deployment Shell:
    ``sudo apt install <deb-dir>/xilinx-u250-xdma-201830.1_<OS>.deb``
    c) Install the Development Shell:
    ``sudo apt install <deb-dir>/xilinx-vcu1525-xdma-201830.1-dev_<OS>.deb``

If you have trouble installing these via the package manager (for example using
a newer distribution like Ubuntu 18.10) it's possible to extract the files and
manually install them. The directory structure of the package mimics the default
install locations on your system, e.g. /opt/xilinx/platforms. If you choose the
extraction route then all you really require for emulation is the files inside
the Development Shell.

The main files required for emulation of a U250 board are found inside the Development Shell under platforms.

The main files required for deployment to a U250 board are inside the Deployment
Shell.

This set of instructions should be applicable to other boards you wish to test,
you can search for boards via the [boards-and-kits](https://www.xilinx.com/products/boards-and-kits/)
page.

## Environment & Setup

For the moment this projects only been tested on Linux (Ubuntu 18.10), so for
now we shall only detail the minimum setup required in this context.

In addition to the required environment variables for the base SYCL
implementation specified in [GetStartedWithSYCLCompiler.md](GetStartedWithSYCLCompiler.md); compilation and
execution of SYCL on FPGAs requires the following:

To setup SDx for access to the `xocc` compiler the following steps are required:

```bash
export XILINX_SDX=/path_to/SDx/2018.3
PATH=$XILINX_SDX/bin:$XILINX_SDX/lib/lnx64.o:$PATH
```

To setup XRT for the runtime the following steps are required:

```bash
export XILINX_XRT=/path_to/xrt
export LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
PATH=$XILINX_XRT/bin:$PATH
```

On top of the above you should specify emulation mode which indicates to the
compiler what it should compile for and to the runtime what mode it should
execute in. It's of note that you will likely encounter problems if the binary
was compiled with a different emulation mode than is currently set in your
environment (the runtime will try to do things it can't).

The emulation mode can be set as:

* `sw_emu` for software emulation, this is the simplest and quickest compilation
  mode that `xocc` provides.
* `hw_emu` for hardware emulation, this more accurately represents the hardware
  your targeting and does more detailed compilation and profiling. It takes
  extra time to compile and link.
* `hw` for actual hardware compilation, takes a significant length of time to
  compile for a specified device target.

The emulation mode can be specified as follows:

```bash
export XCL_EMULATION_MODE=sw_emu
```

Xilinx platform description, your available platforms (device) can be found in
SDx's platform directory. Specifying this tells both compilers the desired
platform your trying to compile for and the runtime the platform it should be
executing for.

```bash
export XILINX_PLATFORM=xilinx_u250_xdma_201830_1
```

Generate an emulation configuration file, this should be in the executable
directory or your path. It's again, important the emulation configuration fits
your compiled binary or you may encounter some trouble. If there is no
configuration file found, it will default to a basic configuration which works
in most cases, but doesn't reflect your ideal platform. XRT warns you in these
cases.

```bash
emconfigutil -f $XILINX_PLATFORM --nd 1
```

## C++ Standard

It's noteworthy that we've altered the SYCL runtime to be compiled using C++20,
most of the current features are C++11 compatible outside of the components in
the Xilinx vendor related directories for the moment. However, this is likely to
change as we're interested in altering the runtime with newer C++ features.

There is an issue at the moment that you can only compile source code with C++11
to C++17 for the time being even though the runtime is compiled using C++20. We
believe it's a side effect of the runtime libraries C++ standard wrapper stubs
needing updated for C++20.

## Compiling a SYCL program

At the moment we only support one step compilation, so you can't easily compile
just the device side component and then link it to the host side component.

The compiler invocation for the `single_task_vector_add.cpp` example inside
the [simple_tests](../test/xocc_tests/simple_tests) folder looks like this:

```bash
$SYCL_BIN_DIR/clang++ -D__SYCL_SPIR_DEVICE__ -std=c++17 -fsycl \
  -fsycl-xocc-device single_task_vector_add.cpp -o single_task_vector_add \
  -lOpenCL
```

Be aware that compiling for FPGA is rather slow.

## Compiler invocation differences

The `-fsycl-xocc-device` compiler directive is a flag we use to force certain
things in the compiler at the moment, like picking our device ToolChain's
Assembler and Linker over the regular SYCL ToolChain's Linker. It also forces
the assembler stage of the Clang compiler to emit LLVM-IR for the moment. In the
future we hope to remove this and have this sort of thing defined by the device
target instead, to be more inline with the main Intel SYCL implementation.

The `__SYCL_SPIR_DEVICE__` environment variable currently tells the runtime to
use SPIR intrinsics in place of SPIR-V intrinsics at the moment,
e.g. `get_global_id` in place of `GlobalInvocationId`. In the future this will
probably be defined by default when `-fsycl-xocc-device` is specified to the
compiler.

## Tested with
* Ubuntu 18.10
* XRT 2018.3
* SDx 2018.3
* Alveo U250 Platform: xilinx_u250_xdma_201830_1
