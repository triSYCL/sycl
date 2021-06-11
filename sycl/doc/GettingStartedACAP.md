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
here: https://gitenterprise.xilinx.com/gauthier/acappp/tree/Merging

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

chroot into an ARM to compile for ARM without cross-compilation (see https://confluence.xilinx.com/display/XRLABS/Running+SYCL+on+Tenzing+Versal+boards#RunningSYCLonTenzingVersalboards-Usingthemfromanx86machine).

``` bash
# go to your home.
cd /home/gauthier

# git clone the repository with libxaiengine
git clone https://gitenterprise.xilinx.com/embeddedsw/embeddedsw.git
# go to libmetal's direcotry
cd embeddedsw/ThirdParty/sw_services/libmetal/src/libmetal

# make a build directory
mkdir -p build
cd build

# configure
cmake ..

# compile
make -j20
# there will be some compilation errors but they are in tests libmetal has been compiled correctly.

# install libmetal
sudo cp lib/libmetal.* /usr/local/lib/
sudo cp lib/include/metal /usr/local/include/

# go to openamp's direcotry
cd ../../../../openamp/src/open-amp/

mkdir -p build
cd build
cmake ..
make -j20
sudo make install

cd ../../../../../../XilinxProcessorIPLib/drivers/aienginev2/

# -D__AIELINUX__ is to have an actual hardware backend not just a testing one.
# -g is to be able to debug the library
make -f Makefile.Linux CFLAGS="-D__AIELINUX__ -g" VERBOSE=1

sudo cp libxaiengine.so.2.1 /usr/local/lib/
sudo ln -sf libxaiengine.so.2.1 /usr/local/lib/libxaiengine.so.2
sudo ln -sf libxaiengine.so.2 /usr/local/lib/libxaiengine.so
sudo cp -r ../include/* /usr/local/include/

```


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

This environnement should be in addition to the normal Vitis one.

Here is an example of a full environement setup script
```bash
XILINX_ROOT=/proj/xbuilds/2020.2_qualified_latest/installs/lin64
VITIS_VERSION=2020.2
HLS_NAME=Vitis
AIE_NAME=cardano

export XILINX_XRT=/storage/gauthier/XRT/build/Debug/opt/xilinx/xrt
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
export PATH=$PATH:$XILINX_SDX/bin:$XILINX_SDX/lib/lnx64.o
source "$XILINX_ROOT/$HLS_NAME/$VITIS_VERSION/settings64.sh" &> /dev/null
source "$XILINX_XRT/xrt/setup.sh" &> /dev/null
TMP_PATH=$(echo $PATH | sed -e 's/$PRIORITY_PATH//g')
export PATH=$PRIORITY_PATH:$TMP_PATH
export EMCONFIG_PATH=/storage/gauthier/
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms/xilinx_u200_xdma_201830_2
export TMP="/storage/gauthier/tmp"

export XILINXD_LICENSE_FILE=2100@aiengine-eng
export LM_LICENSE_FILE=2100@aiengine-eng
export RDI_INTERNAL_ALLOW_PARTIAL_DATA=yes
export AIE_ROOT="$XILINX_ROOT/Vitis/$VITIS_VERSION/$AIE_NAME"
export XILINX_VITIS_AIETOOLS=$AIE_ROOT
export PATH="$PATH:$AIE_ROOT/bin/unwrapped/lnx64.o/"
source $AIE_ROOT/scripts/cardano_env.sh
export CHESSROOT=$AIE_ROOT/tps/lnx64/
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
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

The directory `/net/xsjsycl41/srv/Ubuntu-20.10/arm64-root-server-rw-tmp` we 
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

Create a script called make.sh containing
```bash
CC=$1

DIRNAME=$(dirname $CC)
INCLUDE=$(realpath $DIRNAME/../include/sycl/acap-intrinsic.h)
echo $INCLUDE

shift 1
$CC -std=c++2a -Xclang -fforce-enable-int128 \
  -Xclang -aux-triple -Xclang aarch64-linux-gnu \
  -target aarch64-linux-gnu -mcpu=cortex-a72 -fsycl -nolibsycl \
  -DTRISYCL_XAIE_DEBUG -DTRISYCL_DEBUG -DBOOST_LOG_DYN_LINK \
  -include $INCLUDE -fsycl-unnamed-lambda -ffast-math \
  -fsycl-targets=aie32-xilinx-unknown-sycldevice -g -Wl,-Bdynamic -ftemplate-backtrace-limit=0 \
  -Wl,-rpath=/srv/Ubuntu-20.10/arm64-root-server-rw/usr/lib/aarch64-linux-gnu/ \
  -Wl,/srv/Ubuntu-20.10/arm64-root-server-rw-tmp/lib/aarch64-linux-gnu/libpthread.so \
  -Wl,/srv/Ubuntu-20.10/arm64-root-server-rw/usr/lib/aarch64-linux-gnu/blas/libblas.so.3.9.0 \
  -Wl,/srv/Ubuntu-20.10/arm64-root-server-rw/usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3 \
  `pkg-config gtkmm-3.0 --libs --cflags` `pkg-config opencv4 --libs --cflags`\
  --sysroot /srv/Ubuntu-20.10/arm64-root-server-rw \
  -L/srv/Ubuntu-20.10/arm64-root-server-rw/usr/local/lib/ \
  -lxaiengine -lboost_thread -lboost_log -lboost_log_setup -lboost_context -lboost_fiber -I/storage/gauthier/acap/include $@
```
Replace /storage/gauthier/acap/ by your path to acappp.

Then to compile using:
```bash
# for /path/to/clang++ being the sycl compiler.
./make.sh /path/to/clang++ my_file.cpp
```

For a a simple program that shout compile and run correctly 
you can use the templated model:
```cpp
#include <iostream>
#include <SYCL/sycl.hpp>

using namespace sycl::vendor::xilinx;

template <typename AIE, int X, int Y>
struct prog : acap::aie::tile<AIE, X, Y> {
  void run() {
  }
};

int main() {
  // Define AIE CGRA with all the tiles of a VC1902
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
  // Define an AIE CGRA with all the tiles of a VC1902
  // acap::aie::device<acap::aie::layout::vc1902> d;
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


## Requirements
  * Ubuntu 20.10
  * Vitis 2020.2 with chess for AIE
  
## Disclaimer

This is a WIP document, compiler implementation and library, so it will likely
change rapidly.
