## Disclaimer

This document is incomplete. The compiler and library implementation are experimental, so it will likely
change rapidly or get outdated.

The recent work has mostly focused on the more recent AIE++ library so a large part of this documentation is now outdated.

## Requirements

- Recent Linux distribution;
- Vitis 2022.2 which includes CHESS for AIE.

# AMD AIE Compilation

This document aims to cover the key differences of compiling SYCL for AMD
AIE using the WIP device compiler based on the upstream Intel SYCL compiler.

The main future goal of this SYCL device compiler and SYCL runtime is to target
the AIE PL and APUs, not just the PS and AI Engine cores. The ideal goal is
for all of the various AIE components to be usable seamlessly in a single
source SYCL application. This however is a long term goal. For now we support
AMD FPGAs and AI Engine Cores separately and in an experimental WIP fashion.

## The SYCL Runtime for AIE

The SYCL runtime for AIE is currently not the SYCL library compiled and
packaged with the device compiler in this project. The library packaged with
this project is currently a modified (for AMD FPGA) variation of the Intel
upstream implementation. It's currently only aimed at targeting Intel GPUs,
Intel FPGAs, Intel CPUs, Nvidia GPUs, AMD GPUs and AMD FPGAs.

There are 2 libraries using the same compiler:

- ACAP++, see API in [test](../test/acap/test_aie_mandelbrot.cpp);
- AIE++, see API in [test](../test/aie/mandelbrot.cpp)

which are available from [triSYCL](https://github.com/triSYCL/triSYCL).

The ACAP++ library is usable as a standalone SYCL runtime library without the
device compiler and defaults to a multi-threaded CPU application (all kernels are
executed on the host). This makes it superb for debugging your applications
before trying to target your application on a more complex heterogeneous
architecture like AIE.

The AIE++ library for now only targets AIE hardware as the CPU
implementation is a work-in-progress.

## The SYCL Device Compiler

To compile this compiler follow the [Get Started With SYCL Compiler](sycl/doc/GetStartedWithSYCLCompiler.md)
guide. There is currently no unusual differences in the environment or
compilation arguments required like there is for AMD FPGA. However, you can
equally use the compilation steps for FPGA if you wish.

For some information on what the device compiler actually does please look at
the following documentation:

- [SYCL Compiler and Runtime Design](design/CompilerAndRuntimeDesign.md)
- [SYCL Compiler for AMD FPGA](design/AMD_FPGA_SYCL_compiler_architecture.rst)

The AI Engine compilation flow is slightly different when it comes down to the
nitty gritty details. However, the basic concepts still apply.

But in essence, the device compiler is what will compile your SYCL kernels for a
particular device target and package it into the final binary for consumption
by the SYCL runtime.

## AMD AIEngine Library

chroot into an ARM to compile for ARM without cross-compilation (see https://confluence.xilinx.com/display/XRLABS/Running+SYCL+on+Tenzing+Versal+boards#RunningSYCLonTenzingVersalboards-Usingthemfromanx86machine).

``` bash
# git clone the repository with dependencies of libxaiengine
git clone https://github.com/Xilinx/embeddedsw.git
# go to libmetal's directory
cd embeddedsw/ThirdParty/sw_services/libmetal/src/libmetal

# make a build directory
mkdir -p build
cd build

# configure
cmake ..

# compile
make -j`nproc`
# there will be some compilation errors but they are in tests and libmetal should have been compiled correctly.

# install libmetal
sudo cp lib/libmetal.* /usr/local/lib
sudo cp -r lib/include/metal /usr/local/include

# go to openamp's directory
cd ../../../../openamp/src/open-amp

mkdir -p build
cd build
cmake ..
make -j`nproc`
sudo make install

cd ../../../../../../..

git clone https://github.com/Xilinx/aie-rt.git

cd aie-rt/driver/src

# -D__AIELINUX__ is to have an actual hardware backend not just a fake one.
# -g is to be able to debug the library
make -f Makefile.Linux CFLAGS="-D__AIELINUX__ -g" -j`nproc`

# Adjust verions numbers based on what was inside the git
sudo cp libxaiengine.so.3.0 /usr/local/lib
sudo ln -sf libxaiengine.so.3.0 /usr/local/lib/libxaiengine.so.3
sudo ln -sf libxaiengine.so.3 /usr/local/lib/libxaiengine.so
sudo cp -r ../include/* /usr/local/include
```


## AIE SDF and Chess compilers

To use the SYCL device compiler you will need an installation of the AMD
Vitis framework that contains AIE development environment.

While we do not really use AIE ADF tools, we make use of the tools it
relies on for our AI Engine kernel compilation. In this case the `xchesscc`
compiler which compiles our kernels to an ELF binary loadable by the AI Engine
using Synopsis's `chess` compiler tool-chain.

The following environment setup recipe is specifically for AMD Internal use
and should be modified before this project is open sourced:

```bash
# Generic parameters.
XILINX_ROOT=/proj/xbuilds/2021.2_qualified_latest/installs/lin64
VITIS_VERSION=2021.2
HLS_NAME=Vitis_HLS
AIE_NAME=aietools

# Configuration from parameters.
export XILINXD_LICENSE_FILE=2100@aiengine-eng
export LM_LICENSE_FILE=2100@aiengine-eng
export RDI_INTERNAL_ALLOW_PARTIAL_DATA=yes
export AIE_ROOT="$XILINX_ROOT/Vitis/$VITIS_VERSION/$AIE_NAME"
export XILINX_VITIS_AIETOOLS=$AIE_ROOT
export PATH="$PATH:$AIE_ROOT/bin/unwrapped/lnx64.o/"
source $AIE_ROOT/scripts/cardano_env.sh
export CHESSROOT=$AIE_ROOT/tps/lnx64/
```

This environnement should be in addition to the normal Vitis one.

Here is an example of a full environement setup script
```bash
WORKDIR=$HOME
XILINX_ROOT=/proj/xbuilds/2021.2_qualified_latest/installs/lin64
VITIS_VERSION=2021.2
HLS_NAME=Vitis_HLS
AIE_NAME=aietools

export XILINX_XRT=$WORKDIR/XRT/build/Debug/opt/xilinx/xrt
PRIORITY_PATH=$PATH
export XPTI_TRACE_ENABLE=1
export SYCL_PI_TRACE=1
export SYCL_DEVICE_FILTER="opencl:acc"
export LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$XILINX_XRT/lib:$XILINX_ROOT/Vitis/$VITIS_VERSION/lib/lnx64.o/"
export XILINX_SDX=$XILINX_ROOT/$HLS_NAME/$VITIS_VERSION
export XILINX_VITIS=$XILINX_SDX
export XCL_EMULATION_MODE=sw_emu
export PATH=$PATH:$XILINX_XRT/bin
export XILINX_PLATFORM=xilinx_u200_xdma_201830_2
export PATH=$PATH:$XILINX_SDX/bin:$XILINX_SDX/lib/lnx64.o:$XILINX_ROOT/$HLS_NAME/$VITIS_VERSION/aietools/tps/lnx64/target/bin/LNa64bin
source "$XILINX_ROOT/$HLS_NAME/$VITIS_VERSION/settings64.sh" &> /dev/null
source "$XILINX_XRT/xrt/setup.sh" &> /dev/null
TMP_PATH=$(echo $PATH | sed -e 's,$PRIORITY_PATH,,g')
export PATH=$PRIORITY_PATH:$TMP_PATH
export EMCONFIG_PATH=$WORKDIR/
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms/xilinx_u200_xdma_201830_2
export TMP="$WORKDIR/tmp"

export XILINXD_LICENSE_FILE=2100@aiengine-eng
export LM_LICENSE_FILE=2100@aiengine-eng
export RDI_INTERNAL_ALLOW_PARTIAL_DATA=yes
export AIE_ROOT="$XILINX_ROOT/Vitis/$VITIS_VERSION/$AIE_NAME"
export XILINX_VITIS_AIETOOLS=$AIE_ROOT
export PATH="$PATH:$AIE_ROOT/bin/unwrapped/lnx64.o/"
source $AIE_ROOT/scripts/cardano_env.sh
export CHESSROOT=$AIE_ROOT/tps/lnx64/
```

## Several flavors for compiling a basic AIE++ Hello World

So for these examples we will compile the [hello_world.cpp](sycl/test/acap_tests/hello_world.cpp)
test. It's a 1 processing element (AI Engine core) kernel that prints some
output.

In the following examples we assume the environment variable *SYCL_BIN_DIR* is
setup to point the build/bin directory of the compiled SYCL device compiler.
Although in the two cases where we're only compiling for CPU it's not strictly
required, provided the C++ compiler you use is fairly up-to-date and supports
C++ 20, the commands should work.

The directory `/net/xsjsycl41/srv/Ubuntu-22.04/arm64-root-server-rw-tmp` we  reference to in the cross-compilation commands is an Ubuntu-22.04 arm64
(AKA aarch64) image setup to contain all of the required libraries we need for
cross-compilation.

In all of the following commands we assume you're running the latest
version of Ubuntu or Debian.

### Compiling a basic AIE Hello World for native CPU execution

The following compilation command will compile the `hello_world.cpp` example for
your native CPU. What is meant by native in this case, is whatever architecture
you're compiling on. So compiling on x86, will result in a `hello_world`
binary that executes on an x86 CPU. All kernels will also execute on the CPU,
no device offloading or compilation is done.

```bash
  $SYCL_BIN_DIR/clang++ -std=c++20 hello_world.cpp -I/AIE++/acappp/include \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs`
```
### Cross-compilation for ARM CPU execution

The following compilation command will compile the `hello_world.cpp` example for
ARM CPU. The general idea of the compilation commands is that you're compiling
your binary on an Linux x86 host for an Linux ARM64 (aarch64) target.

In this case the assumption is that the ARM target is the AIE PS which is an
Cortex-A72 CPU, but in theory it could be whatever `-mcpu` target the Clang
compiler supports.

```bash
  $ISYCL_BIN_DIR/clang++ -std=c++20 -target aarch64-linux-gnu -mcpu=cortex-a72 \
    --sysroot /net/xsjsycl41/srv/Ubuntu-19.04/arm64-root-server-rw-tmp \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs` \
    -I/AIE++/acappp/include  hello_world.cpp
```

### Cross-compilation for AIE PS/AI Engine execution

This is the only compilation command that requires the SYCL device compiler.
As the intent is to compile all of the kernels to be offloaded for the device,
and the rest for the host.

From a SYCL perspective the ACAP PS is the host device that will offload to
the other elements of the ACAP AIE architecture. The device in this case is the ACAP AIE
AI Engine cores.

Create a script called `make.sh` containing
```bash
SYCL_TARGET=aie1_32-xilinx-unknown-sycldevice

CC=$1

DIRNAME=$(dirname $CC)
INCLUDE=$(realpath $DIRNAME/../include/sycl/aie-intrinsic.h)

# Edit based on environnement.
AIE_RT_PATH=$WORKDIR/acap
AIE_RT_BUILD_PATH=$WORKDIR/acap
# Location of an Ubuntu root directory
ROOT="/srv/Ubuntu-22.04/arm64-root-server-rw"

shift 1
$CC -std=c++2a -Xclang -fforce-enable-int128 \
  -Xclang -aux-triple -Xclang aarch64-linux-gnu -g \
  -target aarch64-linux-gnu -mcpu=cortex-a72 -fsycl -nolibsycl \
  -DTRISYCL_XAIE_DEBUG -DTRISYCL_DEBUG -DBOOST_LOG_DYN_LINK --sysroot $ROOT \
  -fsycl-unnamed-lambda -I$AIE_RT_BUILD_PATH_deps/experimental_mdspan-src/include/ \
  -fsycl-targets=aie1_32-xilinx-unknown-sycldevice -Wl,-Bdynamic -ftemplate-backtrace-limit=0 \
  -lrt -Wl,-rpath=$ROOT/usr/lib/aarch64-linux-gnu/ -fsycl-mutable-global -Xclang -fsycl-allow-func-ptr \
  -Wl,$ROOT/usr/lib/aarch64-linux-gnu/libpthread.so.0 -ffast-math \
  -Wl,$ROOT/usr/lib/aarch64-linux-gnu/blas/libblas.so.3.9.0 \
  -Wl,$ROOT/usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3 \
  `pkg-config gtkmm-3.0 --libs --cflags` `pkg-config opencv4 --libs --cflags` \
  -L$ROOT/usr/lib/aarch64-linux-gnu/ -Xsycl-target-frontend -fno-exceptions \
  -lxaiengine -lboost_thread -lboost_log -lboost_log_setup -lboost_context -lboost_fiber -I$AIE_RT_PATH/include $@
```
Then to compile using:
```bash
# for /path/to/clang++ being the sycl compiler.
./make.sh /path/to/clang++ my_file.cpp
```

For a a simple program that should compile and run correctly
you can use the templated model:
```cpp
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::vendor::xilinx;

// X and Y are the coordinates of the tile in the architecture
template <typename AIE, int X, int Y>
struct prog : acap::aie::tile<AIE, X, Y> {
  void run() {
    // Some computation
  }
};

int main() {
  // Define an AIE CGRA with 1 tile of a VC1902
  acap::aie::device<acap::aie::layout::size<1,1>> aie;
  // Run up to completion prog on all the tiles
  aie.run<prog>();
}
```

or in the uniform model:
```cpp
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::vendor::xilinx;

int main() {
  // Define an AIE CGRA with 1 tile of a VC1902
  acap::aie::device<acap::aie::layout::size<1, 1>> d;
  //  Submit some work on each tile, which is SYCL sub-device
  d.for_each_tile([](auto& t) {
    /* This will instantiate uniformly the same
       lambda for all the tiles so the tile device compiler is executed
       only once, since each tile has the same code
    */
    t.single_task([&](auto& th) {
    });
  });
}

```
