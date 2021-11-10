Getting started with SYCL with a Xilinx FPGA U200 Alveo board and Ubuntu 21.04
==============================================================================

Disclaimer: nothing here is supported and this is all about a research
project.

We assume you have a Xilinx FPGA U200 Alveo board but it might work
with another board too.

We assume that you have some modern Ubuntu like 21.04 version
installed on an `x86_64` machine. But it might work with other recent
versions of Ubuntu or Debian or even other Linux distributions, with
some adaptations.

:warning: if you are using Linux kernel 5.12+ like shipped with Ubuntu
21.10 or Debian/unstable, you will be hit by the bug
https://github.com/Xilinx/XRT/issues/5943 up to its resolution.  In
the meantime you can always help fixing the bug :-) or install/keep
explicitly an older Linux kernel package and boot on it.

:warning: for some reason Ubuntu 21.10 ships an old version of XRT
which is not to be used here. See [section about
XRT](#installing-the-xilinx-xrt-runtime).


## What's new?

- 2021/06/24: there is a new HLS target along the OpenCL/SPIR compiler
  flow for Xilinx FPGA. The HLS target relies on direct LLVM IR
  feeding and allows finer control by using HLS extensions.

- 2021/10/01: the OpenCL/SPIR device compiler flow has been deprecated
  because it has less features than the HLS device compiler flow and
  we lack resources to maintain both.

## Installing the Alveo U200 board

If you do not have a real board and want to use only software or
hardware emulation, just skip this section.

Install an Alveo U200 board in the machine with the right cooling.
The PCIe auxiliary power is not necessary for simple tests not using the full
power of the board.


### Use a modern BIOS

Update the BIOS of your machine to the latest version. The Alveo U200
board might not be detected by `lspci` at the PCIe level if you do not
have an up-to-date BIOS.

If you are running Linux on top of an EFI BIOS, you can probably use
the firmware capsule concept and try:
```bash
# Install the firmware manager and daemon
sudo apt install fwupdate fwupd
# Refresh the list of available firmware
sudo fwupdmgr refresh
# Do the firmware update if any
sudo fwupdmgr update
```

If you are not running an EFI BIOS, you can follow the manual BIOS
update recipe for your motherboard. Typically look for the
latest BIOS version, put it on a FAT32-formatted USB stick and go
into the BIOS setup at boot time to ask for the explicit
update. Often, there is no need to build a bootable USB stick.


## Installing the Xilinx XRT runtime

```bash
# Get the latest Xilinx runtime. You might try the master branch instead...
# Use either ssh
git clone git@github.com:Xilinx/XRT.git
# or https according to your usual method
git clone https://github.com/Xilinx/XRT.git
cd XRT/build
# Install the required packages
sudo ../src/runtime_src/tools/scripts/xrtdeps.sh
# Compile the Xilinx runtime
./build.sh
# Install the runtime into /opt/xilinx/xrt and compile/install
# the Linux kernel drivers (adapt to the real name if different)
sudo apt install --reinstall ./Release/xrt_202210.2.13.0_21.04-amd64-xrt.deb
```

It will install the user-mode XRT runtime and at least compile and
install the Xilinx device driver modules for the current running kernel,
even if it fails for the other kernels installed on the machine. If
you do not plan to run on a real FPGA board but only use software or
hardware emulation instead, it does not matter if the kernel device
driver is not compiled since it will not be used.

Note that if for some reasons you want to use a debug version of XRT,
use this recipe instead:
```bash
cd Debug
# You need to make explicitly the Debug package because it is not made
# by default
make package
# Install the runtime into /opt/xilinx/xrt and compile/install
# the Linux kernel drivers (adapt to the real name if different)
sudo apt install --reinstall ./xrt_202210.2.13.0_21.04-amd64-xrt.deb
```

:warning: for some reason Ubuntu 21.10 ships an old version of XRT
which is not to be used here. Even if you have installed it like
above, it might be automatically "updated" by some automatic package
updater running on a regular basis. So, if you are running Ubuntu
21.10, to avoid this situation, you can put the
package on hold after the installation with:
```bash
sudo apt-mark hold xrt
```

Check that the FPGA board is detected:
```bash
sudo /opt/xilinx/xrt/bin/xbutil --legacy flash scan -v
```

which should display something similar to
```
---------------------------------------------------------------------
Legacy xbutil is being deprecated, consider moving to next-gen xbutil
---------------------------------------------------------------------
WARNING: The xbutil sub-command flash has been deprecated. Please use the xbmgmt utility with flash sub-command for equivalent functionality.

Card [0000:04:00.0]
    Card type:		u200
    Flash type:		SPI
    Flashable partition running on FPGA:
        xilinx_u200_GOLDEN_5,[SC=INACTIVE]
    Flashable partitions installed in system:	
        xilinx_u200_xdma_201830_2,[ID=0x5d1211e8],[SC=4.2.0]
```

## Install Vitis 2021.1

You need the framework to do the hardware synthesis, taking the SPIR
or LLVM IR intermediate representation generated by the SYCL compiler
(or HLS C++ or OpenCL or...) and generating some FPGA configuration
bitstream.

For this, download from
https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html
the "Xilinx Unified Installer 2021.1 SFD"
somewhere you have enough storage.

Create a `/opt/xilinx` with the right access mode so you can work in
it.

Materialize the installer files with
```bash
tar zxvf .../Xilinx_Unified_2021.1_0610_2318.tar.gz
```

Since the graphics Java installer does not work on modern Linux
distributions like Ubuntu 21.04, use the batch-mode version:
```bash
Xilinx_Unified_2021.1_0610_2318/xsetup --batch Install --location /opt/xilinx --agree 3rdPartyEULA,WebTalkTerms,XilinxEULA --edition "Vitis Unified Software Platform"
```
and select `Vitis`.

Note that if you want to uninstall this later, you can typically use:
```bash
/opt/xilinx/.xinstall/Vitis_2021.1/xsetup --batch Uninstall
```

## Install the target platform for the FPGA board

To execute some kernels on the FPGA board you need a deployment target platform
which is an FPGA configuration that pre-defines an architecture on the
FPGA to execute some kernels in some reconfigurable area.

To develop some kernels for your FPGA board or to run some
simulations, you need a development target platform that will contain
some internal description specific to the FPGA and its deployment
target platform so that the tools can generate the right bitstream for
the kernels or the simulation details.

Pick the latest deployment and development target platforms from
https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
for your board and for you OS. It might be an older version. For
example it is possible to use a `2018.3` target platform with Vitis `2021.1`,
and a version for Ubuntu `18.04` on a more recent version of Ubuntu or
Debian.

Install the target platforms:
```bash
sudo apt install ./xilinx-u200-xdma-201830.2-2580015_18.04.deb
sudo apt install ./xilinx-u200-xdma-201830.2-dev-2580015_18.04.deb
```
from where they have been downloaded or adapt the paths to them.


### Flash the board

If you do not want to use a real board or if you have access to a
machine with a board which is already configured, like an XACC cluster
https://xilinx.github.io/xacc, skip this section.

If you want to install a real board, follow the recipe from
https://www.xilinx.com/support/documentation/boards_and_kits/accelerator-cards/1_9/ug1301-getting-started-guide-alveo-accelerator-cards.pdf
about how to correctly generate the exact flashing command.

Typically you run:
```bash
sudo /opt/xilinx/xrt/bin/xbutil --legacy flash scan
XBFLASH -- Xilinx Card Flash Utility
Card [0]
	Card BDF:		0000:04:00.0
	Card type:		u200
	Flash type:		SPI
	Shell running on FPGA:
		xilinx_u200_GOLDEN_2,[SC=1.8]
	Shell package installed in system:
		xilinx_u200_xdma_201830_2,[ID=0x5d1211e8],[SC=4.2.0]
```
to get the information about the installed target platform and you
translate this into a flashing command according to the parameters
above or just follow the information you got when installing
previously the deployment target platform:
```bash
rkeryell@xsjsycl41:~$ sudo /opt/xilinx/xrt/bin/xbutil flash -a xilinx_u200_xdma_201830_2 -t 1561465320
---------------------------------------------------------------------
Legacy xbutil is being deprecated, consider moving to next-gen xbutil
---------------------------------------------------------------------
WARNING: The xbutil sub-command flash has been deprecated. Please use the xbmgmt utility with flash sub-command for equivalent functionality.

	 Status: shell needs updating
	 Current shell: xilinx_u200_GOLDEN_5
	 Shell to be flashed: xilinx_u200_xdma_201830_2
Are you sure you wish to proceed? [y/n]: y

Updating shell on card[0000:04:00.0]
Bitstream guard installed on flash @0x1002000
Persisted 451594 bytes of meta data to flash 0 @0x7f91bca
Extracting bitstream from MCS data:
............................................
Extracted 45859024 bytes from bitstream @0x1002000
Writing bitstream to flash 0:
............................................
Bitstream guard removed from flash
Successfully flashed Card[0000:04:00.0]

1 Card(s) flashed successfully.
Cold reboot machine to load the new image on card(s).
```

Unfortunately you need to "cold reboot" the machine to have the new
target platform loaded inside the FPGA, which means to really
power-off the machine so the new instantiated PCIe interface of the
card can actually be rediscovered by the host machine (everything is
configurable with an FPGA!).

Then after rebooting, you can check with a pre-compiled FPGA program
provided by the target platform that the board is working correctly
with (the device id below is to adapt to your card):
```bash
sudo /opt/xilinx/xrt/bin/xbutil validate --device 0000:04:00.1
INFO: Found 1 cards

INFO: Validating card[0]: xilinx_u200_xdma_201830_2
INFO: == Starting AUX power connector check: 
INFO: == AUX power connector check PASSED
INFO: == Starting Power warning check: 
INFO: == Power warning check PASSED
INFO: == Starting PCIE link check: 
INFO: == PCIE link check PASSED
INFO: == Starting SC firmware version check: 
INFO: == SC firmware version check PASSED
INFO: == Starting verify kernel test: 
INFO: == verify kernel test PASSED
INFO: == Starting IOPS test: 
Maximum IOPS: 109996 (hello)
INFO: == IOPS test PASSED
INFO: == Starting DMA test: 
Host -> PCIe -> FPGA write bandwidth = 8992.138465 MB/s
Host <- PCIe <- FPGA read bandwidth = 11756.230429 MB/s
INFO: == DMA test PASSED
INFO: == Starting device memory bandwidth test: 
...........
Maximum throughput: 48073 MB/s
INFO: == device memory bandwidth test PASSED
INFO: == Starting PCIE peer-to-peer test: 
P2P BAR is not enabled. Skipping validation
INFO: == PCIE peer-to-peer test SKIPPED
INFO: == Starting memory-to-memory DMA test: 
bank0 -> bank1 M2M bandwidth: 11685.8 MB/s	
bank0 -> bank2 M2M bandwidth: 11736.7 MB/s	
bank0 -> bank3 M2M bandwidth: 11748 MB/s	
bank1 -> bank2 M2M bandwidth: 11630.5 MB/s	
bank1 -> bank3 M2M bandwidth: 11710.9 MB/s	
bank2 -> bank3 M2M bandwidth: 11700.2 MB/s	
INFO: == memory-to-memory DMA test PASSED
INFO: == Starting host memory bandwidth test: 
Host_mem is not available. Skipping validation
INFO: == host memory bandwidth test SKIPPED
INFO: Card[0] validated successfully.

INFO: All cards validated successfully.
```


## Compile the SYCL compiler

Building SYCL can be done with Python scripts:
```
# Pick some place where SYCL has to be compiled, such as:
SYCL_HOME=~/sycl_workspace
mkdir $SYCL_HOME
cd $SYCL_HOME
git clone --branch sycl/unified/master git@github.com:triSYCL/sycl.git llvm
python $SYCL_HOME/llvm/buildbot/configure.py
python $SYCL_HOME/llvm/buildbot/compile.py
```

These scripts have many options which can be displayed when using the
`--help` option. For example to configure with CUDA support, without
treating compiler warnings as errors and producing a compiler database
to be used by tools like LSP server like `clangd`: ``` python
$SYCL_HOME/llvm/buildbot/configure.py --cuda -no-werror
--cmake-opt="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" ``` For more control,
see [section Build](#build).


## Compiling and running a SYCL application

The typical environment is setup with something like
```bash
# The place where SYCL has been compiled:
SYCL_HOME=~/sycl_workspace
# The version of Vitis you want to use
XILINX_VERSION=2021.1
# The target platform for the FPGA board model
export XILINX_PLATFORM=xilinx_u200_xdma_201830_2
# Where all the Xilinx tools are
XILINX_ROOT=/opt/xilinx
# Where the SYCL compiler binaries are:
SYCL_BIN_DIR=$SYCL_HOME/llvm/build/bin
export XILINX_XRT=$XILINX_ROOT/xrt
export XILINX_VITIS=$XILINX_ROOT/Vitis/$XILINX_VERSION
export XILINX_VIVADO=$XILINX_ROOT/Vivado/$XILINX_VERSION
# Add the various tools in the PATH
PATH=$PATH:$SYCL_BIN_DIR:$XILINX_XRT/bin:$XILINX_VITIS/bin:$XILINX_VIVADO/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XILINX_XRT/lib:$XILINX_VITIS/lib/lnx64.o:$SYCL_HOME/llvm/build/lib
# Setup LIBRARY_PATH used in hw and hw_emu mode
# Ask ldconfig about the list of system library directories
export LIBRARY_PATH=$(ldconfig --verbose 2>/dev/null | grep ':$' | tr -d '\n')
# Setup device to be used in simulation mode.
# Instead of running emconfigutil all over the place with
# emconfig.json everywhere, put once at a common place:
export EMCONFIG_PATH=~/.Xilinx
emconfigutil --platform $XILINX_PLATFORM --od $EMCONFIG_PATH --save-temps
```

Optionally,
[configuration files](https://www.xilinx.com/developer/articles/using-configuration-files-to-control-vitis-compilation.html)
for Vitis compiler and linker can be specified in, respectively,
`SYCL_VXX_COMP_CONFIG` and `SYCL_VXX_LINK_CONFIG` environment variables.

You can compile an application either for real FPGA execution,
software emulation (the SYCL device code is executed by the XRT
runtime on CPU) or hardware emulation (the SYCL device code is
synthesized into RTL Verilog and run by an RTL simulator such as
`xsim`).

In addition, two compilation flows for compiling SYCL kernels are
provided:

- a new HLS device compiler flow is now developed, that aims at
  compiling kernels to LLVM bitcode similar to what is produced by the
  open source Xilinx HLS front-end. This way, anything supported by
  Xilinx HLS C++ should be supported at some point in the future;

- the SPIR flow device compiler, the first to have been supported by
  the tool, aiming at using OpenCL C-like features. But it is
  deprecated now since it provides less features than the HLS one.

Note that the software and hardware emulation might not work for some
system incompatibility reasons because Vitis comes with a lot of
system-specific assumptions with a lot of old compilers and libraries
instead of just using the ones from the system and the mix-and-match
might be wrong on your current system... But the hardware execution
just requires the open-source XRT that should have been compiled just
using what is available on the system.

Architecture provided to the `sycl-targets` Clang flag selects the
compilation mode. Supported architectures are:

|                                    | Software simulation | Hardware emulation  | Hardware        |
|------------------------------------|---------------------|---------------------|-----------------|
| SPIR compilation flow (deprecated) | `fpga64_sw_emu`     | `fpga64_hw_emu`     | `fpga64_hw`     |
| HLS compilation flow               | Unsupported yet     | `fpga64_hls_hw_emu` | `fpga64_hls_hw` |

Only one `fpga64_*` architecture is allowed in the `sycl-targets`
flag.

The SYCL HLS compilation flow does not support software emulation because
of internal Xilinx issue https://jira.xilinx.com/browse/CR-1099885
But as SYCL allows also execution on a CPU device, it can replace the
back-end software emulation.


### Small examples

To run an example from the provided examples:

- with hardware emulation:
  ```bash
  cd $SYCL_HOME/llvm/sycl/test/on-device/xocc/simple_tests
  # Instruct the compiler and runtime to use FPGA hardware emulation with HLS flow
  # Compile the SYCL program down to a host fat binary including the RTL for simulation
  $SYCL_BIN_DIR/clang++ -std=c++20 -fsycl -fsycl-targets=fpga64_hls_hw_emu \
    single_task_vector_add.cpp -o single_task_vector_add
  # Run the hardware emulation
  ./single_task_vector_add
  ```

- with real hardware execution on FPGA:
  ```bash
  cd $SYCL_HOME/llvm/sycl/test/on-device/xocc/simple_tests
  # Instruct the compiler to use real FPGA hardware execution with HLS flow
  # Compile the SYCL program down to a host fat binary including the FPGA bitstream
  $SYCL_BIN_DIR/clang++ -std=c++20 -fsycl -fsycl-targets=fpga64_hls_hw \
    single_task_vector_add.cpp -o single_task_vector_add
  # Run on the real FPGA board
  ./single_task_vector_add
  ```
Note that only the flag `-fsycl-targets` is changed across the previous examples.


### Looking at the FPGA layout with Vivado

SYCL for Vitis can generate a lot of files, report files, log
files... including Xilinx `.xpr` projects which can be used by Vivado
for inspection, for example to look at the physical layout of the FPGA.

For this, you need to compile with an environment variable stating
that the temporary files have to be kept, for example with:
```bash
export SYCL_VXX_KEEP_CLUTTER=True
```

Then, after compiling, you will have in your temporary directory
(typically `/tmp`) a directory with a name related to the
binary you have built, like `answer_42-bdb894rl239zxg`.

In this directory you can find report files in various textual or HTML
format per kernel in the `vxx_comp_report` directory giving
information about expected frequency operation and FPGA resource usage.

In the `vxx_link_report/link` there is information after linking all
the kernel together.

More interesting there is in
`vxx_link_tmp/link/vivado/vpl/prj/prj.xpr` the project file which can
be opened with Vivado to look at the FPGA schematics and layout.


### Running the test suite

Selecting the target for which the tests are run is done using
the `VXX_TARGET` environment variable. It defaults to `hls_hw_emu`.
The value to give is the same as the associated sycl target, with
the `fpga64_` prefix trimmed. Namely:

|                       | Software simulation | Hardware emulation  | Hardware        |
|-----------------------|---------------------|---------------------|-----------------|
| SPIR compilation flow | `sw_emu`     | `hw_emu`     | `hw`     |
| HLS compilation flow  | Unsupported yet | `hls_hw_emu` | `hls_hw` |

Note that the SPIR compilation flow has been discontinued.

- Run the `xocc` test suite with hardware emulation (HLS flow):
  ```bash
  cd $SYCL_HOME/llvm/build
  export VXX_TARGET=hls_hw_emu
  cmake --build . --parallel `nproc` --target check-sycl-xocc-j4
  ```
  This takes usually 15-30 minutes with a good CPU.

- Run the `xocc` test suite with real hardware execution on FPGA (HLS flow):
  ```bash
  cd $SYCL_HOME/llvm/build
  export VXX_TARGET=hls_hw
  cmake --build . --parallel `nproc` --target check-sycl-xocc-j4
  ```
  This takes usually 8+ hours.

`check-sycl-xocc-jmax` will run the tests on as many cores as is
available on the system. But for `hw` and `hw_emu` execution mode,
this usually means the system will run out of RAM even with 64G so
`check-sycl-xocc-j4` should be used to run only 4 tests in
parallel. There is also a `j2` version to use only 2 cores.

To launch the compilation on all the SYCL tests, not only the `xocc`
ones, there are the targets `check-sycl-all-jmax`,
`check-sycl-all-j2` and `check-sycl-all-j4`


### Running a bigger example on real FPGA

To run a SYCL translation of
https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/edge_detection
```bash
cd $SYCL_HOME/llvm/sycl/test/on-device/xocc/edge_detection
# Instruct the compiler and runtime to use real FPGA hardware execution
$SYCL_BIN_DIR/clang++ -std=c++20 -fsycl \
    -fsycl-targets=fpga64_hls_hw edge_detection.cpp \
    -o edge_detection `pkg-config --libs --cflags opencv4`
# Execute on one of the images
./edge_detection data/input/eiffel.bmp
```
and then look at the `input.bmp` and `output.bmp` images.

There is another application along using a webcam instead, if you have one
on your machine.


## Cleaning up some buffer allocation

The XRT memory model is richer (and more complex...) than the
OpenCL memory model: buffers can be allocated on some DDR or HBM
memory banks, buffers can be shared between different processes on the
host, etc.

This means that the buffer lifetime is actually handled by the `xocl`
kernel driver across the Linux system image to manage this memory
sharing across different processes, if required. The OpenCL buffer
creation and destruction APIs handle this and fortunately this is
hidden by the higher-level SYCL framework.

But if a SYCL program crashes before freeing the OpenCL buffer
and the user tries to allocate some other buffers at the same place on
the FPGA board with another program, then the runtime refuses to load
the program, with some error like:
```
[XRT] ERROR: Failed to load xclbin.
OpenCL API failed. /var/tmp/rkeryell/SYCL/sycl/sycl/source/detail/program_manager/program_manager.cpp:78: OpenCL API returns: -44 (CL_INVALID_PROGRAM)
```
and with some kernel message that can be displayed by executing `dmesg`
like:
```
[256564.482271] [drm] Finding MEM_TOPOLOGY section header
[256564.482273] [drm] Section MEM_TOPOLOGY details:
[256564.482274] [drm]   offset = 0x29e5908
[256564.482275] [drm]   size = 0x120
[256564.482282] xocl 0000:04:00.1: xocl_check_topology: The ddr 0 has pre-existing buffer allocations, please exit and re-run.
[256564.482287] xocl 0000:04:00.1: xocl_read_axlf_helper: err: -1
```

Then you need to free the buffer explicitly because the device
driver still has the hope a program wants to use the data of the
allocated buffer in the future...

This can be done by removing the kernel driver and reloading it by
executing:
```bash
sudo rmmod xocl
sudo modprobe xocl
```

## Xilinx Macros

``__SYCL_XILINX_SW_EMU_MODE__`` will be defined when compiling device code in sw_emu mode

``__SYCL_XILINX_HW_EMU_MODE__`` will be defined when compiling device code in hw_emu mode

``__SYCL_XILINX_HW_MODE__`` will be defined when compiling device code in hw mode

when compiling host code none of the ``__SYCL_XILINX_*_MODE__`` macros will be defined.

``__SYCL_HAS_XILINX_DEVICE__`` will be defined on the host if one of the specified targets is a Xilinx device or on a Xilinx device

## Xilinx FPGA SYCL compiler architecture

[Architecture of the Xilinx SYCL
compiler](Xilinx_sycl_compiler_architecture.rst) describes the
compiler architecture.

This document aims to cover the key differences of compiling SYCL for Xilinx
FPGAs. Things like building the compiler and library remain the same but other
things like the compiler invocation for Xilinx FPGA compilation is a little
different. As a general rule of thumb we're trying to keep things as close as we
can to the Intel implementation, but in some areas were still working on that.

One of the significant differences of compilation for Xilinx FPGAs over the
ordinary compiler directive is that Xilinx devices require offline compilation
of SYCL kernels to binary before being wrapped into the end fat binary. The
offline compilation of these kernels is done by Xilinx's `v++` compiler rather
than the SYCL device compiler itself in this case. The device compiler's job is
to compile SYCL kernels to a format edible by `v++`, then take the output of
`v++` and wrap it into the fat binary as normal.

The current Intel SYCL implementation revolves around SPIR-V while
Xilinx's `v++` compiler can only ingest plain LLVM IR 6.x or LLVM IR
6.x with a SPIR-df flavor as an intermediate representation. SPIR-df
is some LLVM IR with some SPIR decorations. It is similar to the
SPIR-2.0 provisional specification but does not requires the LLVM IR
version to be exactly 3.4. It uses just the encoding of the LLVM used,
which explains the `-df` as "de-facto". This is the term used in the
team to present this non-conforming and non-standard SPIR.

So a lot of our modifications revolve
around being the middle man between `v++` and the SYCL device
compiler and runtime for the moment, they are not the simple whims of
the insane! Hopefully...


### Extra Notes:

* The Driver ToolChain, currently makes some assumptions about the
  `Vitis` installation. For example, it assumes that `v++` is inside
  Vitis's `bin` folder and that the `lib` folder containing `SPIR`
  builtins that kernels are linked against are in a `lnx64/lib`
  directory relative to the `bin` folder parent.

## Debugging the SYCL implementation

### Build

For serious work on the SYCL toolchain it is better to
not use the scripts in `$SYCL_HOME/buildbot` but to invoke `cmake` directly.
This gives much more control over the build configuration.

It is quite useful to work with two builds, a Release one used by default
and a Debug one used for debugging.

Note: the configuration of environment variables must be done before the `cmake` invocation.

A possible Release configuration and build script targeting Xilinx FPGA, CUDA & OpenCL:
```bash
cd $SYCL_HOME
mkdir -p "build-Release" && cd "build-Release" && cmake \
 -DCMAKE_INSTALL_PREFIX="../Install-Release" \
 -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -fno-omit-frame-pointer" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
 -G Ninja \
 -DCMAKE_C_COMPILER="/usr/bin/clang-13" \
 -DCMAKE_CXX_COMPILER="/usr/bin/clang++-13" \
 -DLLVM_USE_LINKER="lld-13" \
 -DCMAKE_BUILD_TYPE=Release \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
 -DLLVM_EXTERNAL_PROJECTS="sycl;llvm-spirv;opencl-aot;xpti;libdevice" \
 -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/sycl \
 -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm-spirv \
 -DLLVM_EXTERNAL_XPTI_SOURCE_DIR=$SYCL_HOME/xpti \
 -DLLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR=$SYCL_HOME/libdevice \
 -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl;opencl-aot;xpti;libdevice;libclc" \
 -DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl" \
 -DSYCL_BUILD_PI_CUDA=ON \
 -DLLVM_BUILD_TOOLS=ON \
 -DSYCL_INCLUDE_TESTS=ON \
 -DLLVM_ENABLE_DOXYGEN=OFF \
 -DLLVM_ENABLE_SPHINX=OFF \
 -DBUILD_SHARED_LIBS=ON \
 -DSYCL_ENABLE_XPTI_TRACING=ON \
 $SYCL_HOME/llvm
```
then build with
```bash
cd $SYCL_HOME
ninja -C build-Release sycl-toolchain
```

A possible Debug configuration and build script targeting Xilinx FPGA, CUDA & OpenCL:
```bash
cd $SYCL_HOME
mkdir -p "build-Debug" && cd "build-Debug" && cmake \
 -DCMAKE_BUILD_TYPE="Debug" \
 -DCMAKE_CXX_FLAGS_DEBUG="-g -fstandalone-debug" \
 -DLLVM_TABLEGEN="$(pwd)/../build-Release/bin/llvm-tblgen" \
 -DCLANG_TABLEGEN="$(pwd)/../build-Release/bin/clang-tblgen" \
 -DCMAKE_INSTALL_PREFIX="../Install-Debug" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
 -G Ninja \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DCMAKE_C_COMPILER="/usr/bin/clang-13" \
 -DCMAKE_CXX_COMPILER="/usr/bin/clang++-13" \
 -DLLVM_USE_LINKER="lld-13" \
 -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
 -DLLVM_EXTERNAL_PROJECTS="sycl;llvm-spirv;opencl-aot;xpti;libdevice" \
 -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/sycl \
 -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm-spirv \
 -DLLVM_EXTERNAL_XPTI_SOURCE_DIR=$SYCL_HOME/xpti \
 -DLLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR=$SYCL_HOME/libdevice \
 -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl;opencl-aot;xpti;libdevice;libclc" \
 -DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl" \
 -DSYCL_BUILD_PI_CUDA=ON \
 -DLLVM_BUILD_TOOLS=ON \
 -DSYCL_INCLUDE_TESTS=ON \
 -DLLVM_ENABLE_DOXYGEN=OFF \
 -DLLVM_ENABLE_SPHINX=OFF \
 -DBUILD_SHARED_LIBS=ON \
 -DSYCL_ENABLE_XPTI_TRACING=ON \
 $SYCL_HOME/llvm
```
then build with
```bash
cd $SYCL_HOME
ninja -C build-Debug sycl-toolchain
```

* `-CMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -fno-omit-frame-pointer"` to have an optimized build with relatively accurate stack traces;
* `-G Ninja` use `ninja` instead of `make` to speedup builds;
* `-DLLVM_USE_LINKER="lld-13"` select `lld` or `gold` instead of `ld` to speedup builds;
* `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"` to build only the targets needed, like NVPTX for CUDA support;
* `-DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl"` needed for CUDA support;
* `-DSYCL_ENABLE_XPTI_TRACING=ON` adds useful debugging capabilities;
* `-DBUILD_SHARED_LIBS=ON` to speed up the build process by using shared libraries;
* `-DLLVM_TABLEGEN="$SYCL_HOME/build-Release/bin/llvm-tblgen" -DCLANG_TABLEGEN="$SYCL_HOME/build-Release/bin/clang-tblgen"` to reuse the tablegen built with the Release version and speedup builds. The Release build must be done before configuring the Debug one;
* `-DCMAKE_CXX_FLAGS_DEBUG="-g -fstandalone-debug"` to have full debug information.

For details about the `CMake` configuration see https://llvm.org/docs/CMake.html

While building the `sycl-toolchain` target, the device runtime for `spirv` and `nvptx` targets gets compiled using the device compiler from the specific build.
Building anything with a debug compiler is very slow, but it can be speedup using:
```bash
export LD_LIBRARY_PATH=$SYCL_HOME/build-Release/lib:$LD_LIBRARY_PATH
```

This will make the debug compiler select the dynamic libraries of the release compiler and speedup the build. This only works because the Release and Debug build have ABI compatible configuration, changing `LLVM_ENABLE_ASSERTIONS` or other configuration may change that.

### Debugging

#### Debugging the driver and intermediate steps

During the compilation, an temporary directory is created in which all Vitis inputs, commands, outputs and logs are stored.
By default, this directory is deleted as soon as the compilation ends (even when it fails).

In order to keep it, set the `SYCL_VXX_KEEP_CLUTTER` environment variable to True.

The compiler will output a similar to

```
Temporary clutter in /tmp/EXECNAME-e5ece1pxk5rz43 will not be deleted
```

Informing you of where those files are kept (`/tmp/EXECNAME-e5ece1pxk5rz43` in this case).

#### Environnement variables

Some environment variables are very useful for debugging:

```bash
# Redirect the directory used as temporary directory
# for the compiler and various tools
export TMP=$SYCL_HOME/../tmp

# SYCL_PI_TRACE should always be at least at 1, this make the SYCL runtime emit logs about which device is selected
export SYCL_PI_TRACE=1

# SYCL_PI_TRACE can be set at -1 to have full debug information but this is quite noisy
export SYCL_PI_TRACE=-1
```

#### Clang flags

Some useful Clang flags:
* `-ccc-print-phases` outputs the graph of compilation phases;
* `-ccc-print-bindings` outputs the mapping from compilation phases to commands;
* `-###` outputs the commands to run to compile. The driver will create files to reserve them for those commands.

#### Running a single test

To run a test from the test suite in isolation, use:
```bash
/path/to/build/dir/bin/llvm-lit -v --param XOCC=all path/to/test.cpp
```
where all tests utilities must have been build for this to work.

#### v++ Logs

The kinds of following errors are typical of a back-end issue:
```
ERROR: [v++ 60-300] Failed to build kernel(ip) kernel_name, see log for details: [...]/vitis_hls.log
ERROR: [v++ 60-599] Kernel compilation failed to complete
ERROR: [v++ 60-592] Failed to finish compilation
```
the path `[...]` contains a hash and kernel names.

Please follow the log chain to identify the cause of this issue. 
Keep in mind that if the `SYCL_VXX_KEEP_CLUTTER` environment variable is not set, 
log files will be deleted as soon as the compilation process exit, meaning that they are 
probably already gone when you get this error message.

#### llvm-reduce

It is possible to use `llvm-reduce` to track down `v++` issues.
First build `llvm-reduce` with:
```bash
ninja -C build-Release llvm-reduce
```
then build a script called later `is_interesting_llvm.sh` to exhibits the `v++` bug.
This will look like the following:
```bash
#!/bin/bash

# Downgrade the IR generate by llvm-reduce
./build-Release/bin/opt -verify -S -xoccIRDowngrader $1 -o $1.tmp.ll || exit 1

# Assemble the IR using Vitis's assembler
.../clang-3.9-csynth/bin/llvm-as $1.tmp.ll -o $1.tmp.xpirbc

# run the command that crashed v++ on the file generated by llvm-reduce while outputting everything to a file
.../clang-3.9-csynth/bin/clang ... -x ir $1.tmp.xpirbc -o - &> out

# Save the exit code of the command since it might have crashed
res=$?

# Test that the output matches the original error
cat out | grep "..." > /dev/null || exit 1

# Test the error code
if [ $res -eq 254 ]; then
# exit 0 to indicate this still exhibits the original bug
  exit 0
else
# exit 1 to indicate this doesn't exhibit the original bug
  exit 1
fi
```
then we can run `llvm-reduce` by doing the following:
```bash
# Disassemble the file that crashed Vitis's Clang
./build-Release/bin/opt -S file_that_crashed_vitis_clang.bc -o tmp.ll

# Run llvm-reduce
./build-Release/bin/llvm-reduce --test=./is_interesting_llvm.sh tmp.ll
```

When `llvm-reduce` finishes, the reduced IR crashing `v++` will be in a file named `reduced.ll`.
