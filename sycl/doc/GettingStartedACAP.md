# Xilinx ACAP Compilation

This document aims to cover the key differences of compiling SYCL for Xilinx
ACAP using the WIP device compiler based on the upstream Intel SYCL compiler.

The main future goal of this SYCL device compiler and SYCL runtime is to target 
the ACAP PL and APUs, not just the PS and AI Engine cores. The ideal goal is
for all of the various ACAP components to be usable seamlessly in a single
source SYCL application. This however is a long term goal. For now we support 
Xilinx FPGAs and AI Engine Cores separately and in an experimental WIP fashion.

## The SYCL Runtime for ACAP

The SYCL runtime for ACAP is currently not the SYCL library compiled and 
packaged with the device compiler in this project. The library packaged with 
this project is currently a modified (for Xilinx FPGA) variation of the Intel 
upstream implementation. It's currently only aimed at targeting Intel 
architectures and Xilinx FPGA.

The runtime you'll require for now is the ACAP++ library for AI Engine found 
here: https://gitenterprise.xilinx.com/rkeryell/acappp

In particular you will require the `acap-device` branch, which is a WIP branch 
for ACAP AI Engine compilation. This library is a header only SYCL library with 
a large number of extensions to SYCL for ACAP AI Engine.

The ACAP++ library is usable as a standalone SYCL runtime library without the
device compiler it defaults to a multi-threaded CPU application (all kernels are 
executed on the host). This makes it superb for debugging your applications 
before trying to target your application on a more complex heterogeneous 
architecture like ACAP.

## The SYCL Device Compiler

To compile this compiler follow the [Get Started With SYCL Compiler](sycl/doc/GetStartedWithSYCLCompiler.md)
guide. There is currently no unusual differences in the environment or 
compilation arguments required like there is for Xilinx FPGA. However, you can 
equally use the compilation steps for FPGA if you wish.

For some information on what the device compiler actually does please look at
the following documentation:
- [SYCL Compiler and Runtime Design](sycl/doc/SYCL_compiler_and_runtime_design.md)
- [SYCL Compiler for Xilinx FPGA](sycl/doc/doc_sycl_compiler.rst)

The AI Engine compilation flow is slightly different when it comes down to the 
nitty gritty details. However, the basic concepts still apply.

But in essence the device compiler, is what will compile your SYCL kernels for a
particular device target and package it into the final binary for consumption
by the SYCL runtime.

## Xilinx AIEngine Library

TODO: Add description and installation/setup instructions for libxaiengine 

## Cardano and Chess

To use the SYCL device compiler you will need an installation of the Xilinx 
Scout tool that contains Cardano.

While we don't really have any interest in Cardano, we make use of the tools it
relies on for our AI Engine kernel compilation. In this case the `xchesscc` 
compiler which compiles our kernels to an ELF binary loadable by the AI Engine 
using Synopsis's `chess` compiler toolchain.

The following environment setup recipe is specifically for Xilinx Internal use
and should be modified before this project is open sourced:

```bash
export XILINXD_LICENSE_FILE=2100@aiengine-eng
export LM_LICENSE_FILE=2100@aiengine-eng
export RDI_INTERNAL_ALLOW_PARTIAL_DATA=yes

declare RELEASE="HEAD"
declare KIND="daily_latest"

export PATH=/proj/xbuilds/${RELEASE}_${KIND}/installs/lin64/Scout/$RELEASE/cardano/bin/unwrapped/lnx64.o/:$PATH
export CARDANO_ROOT="/proj/xbuilds/${RELEASE}_${KIND}/installs/lin64/Scout/$RELEASE/cardano"
source $CARDANO_ROOT/scripts/cardano_env.sh

export CHESSROOT=/proj/xbuilds/${RELEASE}_${KIND}/installs/lin64/Scout/$RELEASE/cardano/tps/lnx64/
```

## Several flavors for compiling a basic ACAP++ Hello World

So for these examples we will compile the [hello_world.cpp](sycl/test/acap_tests/hello_world.cpp) 
test. It's a 1 processing element (AI Engine core) kernel that prints some 
output.

In the following examples we assume the environment variable *SYCL_BIN_DIR* is
setup to point the build/bin directory of the compiled SYCL device compiler. 
Although in the two cases where we're only compiling for CPU it's not strictly 
required, provided the C++ compiler you use is fairly up-to-date and supports 
C++ 2a, the commands should work.

The directory `/net/xsjsycl41/srv/Ubuntu-19.04/arm64-root-server-rw-tmp` we 
make reference to in the cross-compilation commands is an Ubuntu-19.04 arm64 
(AKA aarch64) image setup to contain all of the required libraries we need for 
cross-compilation.

In all of the following commands we assume you're running the latest
version of Ubuntu Debian.

### Compiling a basic ACAP Hello World for native CPU execution

The following compilation command will compile the hello_world.cpp example for 
your native CPU. What is meant by native in this case, is whatever architecture 
you're compiling on. So compiling on x86, will result in a hello_world 
binary that executes on an x86 CPU. All kernels will also execute on the CPU,
no device offloading or compilation is done.

```bash
  $SYCL_BIN_DIR/clang++ -std=c++2a hello_world.cpp -I/ACAP++/acappp/include \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs`
```
### Cross-compilation for ARM CPU execution

The following compilation command will compile the hello_world.cpp example for 
ARM CPU. The general idea of the compilation commands is that you're compiling 
your binary on an Linux x86 host for an Linux ARM64 (aarch64) target.
 
In this case the assumption is that the ARM target is the ACAP PS which is an 
Cortex-A72 CPU, but in theory it could be whatever `-mcpu` target the Clang 
compiler supports. 
 
```bash
  $ISYCL_BIN_DIR/clang++ -std=c++2a -target aarch64-linux-gnu -mcpu=cortex-a72 \
    --sysroot /net/xsjsycl41/srv/Ubuntu-19.04/arm64-root-server-rw-tmp \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs` \
    -I/ACAP++/acappp/include  hello_world.cpp
```

### Cross-compilation for ACAP PS/AI Engine execution

This is the only compilation command that requires the SYCL device compiler. 
As the intent is to compile all of the kernels for the device to be offloaded, 
and the rest for the host.

From a SYCL perspective the ACAP PS is the host device that will offload to
the other elements of the ACAP architecture. The device in this case is the ACAP 
AI Engine cores.

```bash
  $ISYCL_BIN_DIR/clang++ -std=c++2a -Xclang -fforce-enable-int128 \
  -Xclang -aux-triple -Xclang aarch64-linux-gnu \
    -target aarch64-linux-gnu -mcpu=cortex-a72 -fsycl \
    -fsycl-targets=aie32-xilinx-unknown-sycldevice \
    -fsycl-header-only-library \
    --sysroot /net/xsjsycl41/srv/Ubuntu-19.04/arm64-root-server-rw-tmp \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs` \
    -I/ACAP++/acappp/include  hello_world.cpp
```

Note: This test is nonsensical on real AI Engine hardware at the moment as 
standard output readable by the PS doesn't exist yet. The intent for now is to
showcase the compilation process and test it using some AI Engine simulation.
A more complex hello world for AI Engine hardware is in progress.

## Requirements
  * Ubuntu 19.04
  * Scout 2019.2 with Cardano
  
## Disclaimer

This is a WIP document, compiler implementation and library, so it will likely
change rapidly.
