
# Table of content

1. [Table of content](#Table-of-content)
1. [Introduction](#Introduction)
    1. [What's new?](#Whats-new)
1. [Installation](#Installation)
    1. [Installing the Alveo U200 board](#Installing-the-Alveo-U200-board)
        1. [Use a modern BIOS](#Use-a-modern-BIOS)
    1. [Install Vitis 2022.2](#Install-Vitis-20222)
    1. [Boot on a specific kernel](#Boot-on-a-specific-kernel)
    1. [Installing the AMD XRT runtime](#Installing-the-AMD-XRT-runtime)
    1. [Install the target platform for the FPGA board](#Install-the-target-platform-for-the-FPGA-board)
        1. [Flash and test the board](#Flash-and-test-the-board)
    1. [Compile the SYCL compiler](#Compile-the-SYCL-compiler)
1. [Usage](#Usage)
    1. [Compiling and running a SYCL application](#Compiling-and-running-a-SYCL-application)
        1. [Picking the right device](#Picking-the-right-device)
        1. [Small examples](#Small-examples)
        1. [Looking at the FPGA layout with Vivado](#Looking-at-the-FPGA-layout-with-Vivado)
        1. [Running the test suite](#Running-the-test-suite)
        1. [Running a bigger example on real FPGA](#Running-a-bigger-example-on-real-FPGA)
    1. [Cleaning up some buffer allocation](#Cleaning-up-some-buffer-allocation)
    1. [AMD FPGA extension](#AMD-FPGA-extension)
        1. [Pipelining](#Pipelining)
        1. [Dataflow decorators](#Dataflow-decorators)
        1. [Loop unrolling](#Loop-unrolling)
        1. [Pinning allocators to specific memory banks](#Pinning-allocators-to-specific-memory-banks)
        1. [Array partitioning](#Array-partitioning)
        1. [Multiple annotations](#Multiple-annotations)
    1. [AMD Macros](#AMD-Macros)
    1. [Known issues](#known-issues)
      1. [xsim](#xsim)
      1. [Shared library](#shared-lib)
1. [Implementation](#Implementation)
    1. [AMD FPGA SYCL compiler architecture](#AMD-FPGA-SYCL-compiler-architecture)
    1. [Extra Notes](#Extra-Notes)
    1. [Debugging the SYCL implementation](#Debugging-the-SYCL-implementation)
        1. [Debugging the driver and intermediate steps](#Debugging-the-driver-and-intermediate-steps)
        1. [Debugging the SYCL runtime](#Debugging-the-SYCL-runtime)
        1. [Environnement variables](#Environnement-variables)
        1. [Clang flags](#Clang-flags)
        1. [Running a single test](#Running-a-single-test)
        1. [v++ Logs](#v-Logs)
        1. [llvm-reduce](#llvm-reduce)

# Introduction

Disclaimer: nothing here is supported and this is all about a research
project.

This document is about the normal SYCL single-source compilation flow.
There is also another document about the [C++20 non-single source Vitis
IP mode compilation flow](AMDGettingStartedFPGAIPBlockDesign.md).

We assume you have an AMD FPGA U200 Alveo board but it might work
with another board too.

We assume that you have some modern Ubuntu like 22.04 version
installed on an `x86_64` machine. But it might work with other recent
versions of Ubuntu or Debian or even other Linux distributions, with
some adaptations.

## What's new?

- 2023/05/24:
  - add partial profilling support to `pi_xrt`;
  - clarify `sycl/test/vitis/edge_detection/edge_detection.cpp` test;
  - fix `ONEAPI_DEVICE_SELECTOR` usage syntax;
  - fix documentation for Vitis 2022.2;
  - update to new test layout.

- 2022/12/21:
  - support for Vitis 2022.2 and oneAPI from up-stream;
  - update documentation to Vitis 2022.2;
  - various fixes to enable more backends while using AMD FPGA;
  - add `xrt` support for new `ONEAPI_DEVICE_SELECTOR` environment variable;
  - documentation updated to use `DPCPP_HOME` instead of `SYCL_HOME`;
  - enable ROCm/HIP for AMD GPU in documentation;
  - new code sample using 5 accelerators;
  - fix some bugs in parallel_for emulation for FPGA with HLS flow that interferes with other backends;
  - allows OpenCL accelerator to handle either Intel or AMD FPGA;
  - new API to express several AMD FPGA annotations at once;
  - disable AMD OpenCL for FPGA when XRT for FPGA is enabled to avoid XRT bug;
  - do not propagate `-fPIC` option to AMD FPGA device compiler;
  - remove `--sycl-vxx`;
  - remove unreliable hack `terminate_xsimk` around AMD FPGA emulation
    termination bug;
  - move from SYCL 1.2.1 `CL/sycl.hpp` to SYCL 2020 `sycl/sycl.hpp`;
  - clean up XRT PI;
  - asynchronous kernel launch in XRT backend;
  - support for pipes;
  - fix bug in `memset` lowering;
  - fix bug in `memcpy` unrolling;
  - fix tests.

- 2022/05/23:
  - new XRT backend plugin to use AMD FPGA without the OpenCL layer;
  - SYCL interoperability with XRT backend;
  - add a new HLS-like non-single-source compiler flow relying on the
    C++20 features of the SYCL device-compiler to generate Vitis IP;
  - enable more C++ standard library in Vitis IP mode;
  - new `--vitis-ip-part` option to specify the FPGA target in Vitis
    IP mode;
  - new accessor property to handle HBM bank allocation;
  - remove default allocation on DDR bank 0;
  - allow N-dimensional partitioned arrays;
  - fix bugs in `parallel_for` emulation for FPGA with HLS flow;
  - updated for latest Ubuntu 22.04 Linux version;
  - updated for Vitis 2022.2 and latest XRT;
  - improved documentation;
  - simplify test targets;
  - fix many other bugs;
  - lot of cleanup and refactoring;
  - merge from upstream.

- 2022/02/21:
  - provide `parallel_for` emulation with loop-nest in `single_task` for
    AMD FPGA LLVM IR HLS workflow;
  - updated for latest Alveo DFX platforms like the
    `xilinx_u200_gen3x16_xdma_1_202110_1`;
  - remove some FPGA-specific optimizations breaking HLS compilation
    flow;
  - clean up static unroll;
  - updated for latest Ubuntu 21.10 Linux version;
  - updated for Vitis 2021.2 and prepare for next Vitis release;
  - simplify some examples like the generic executor;
  - work-around various Vitis HLS, XRT & platform bugs;
  - execute Vitis in a process namespace to avoid process left-over;
  - merge from upstream Intel oneAPI DPC++ SYCL implementation;
  - mention 2022 year bug work-around;
  - Xilinx is now AMD after acquisition by AMD, so rename Xilinx mentions.

- 2021/10/01: the OpenCL/SPIR device compiler flow has been deprecated
  because it has less features than the HLS device compiler flow and
  we lack resources to maintain both.

- 2021/06/24: there is a new HLS target along the OpenCL/SPIR compiler
  flow for AMD FPGA. The HLS target relies on direct LLVM IR
  feeding and allows finer control by using HLS extensions.

# Installation

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


## Install Vitis 2022.2

You need the framework to compile some XRT MicroBlaze firmware, taking
the SPIR or LLVM IR intermediate representation generated by the SYCL
compiler (or HLS C++ or OpenCL or...), do the hardware synthesis, and
generating some FPGA configuration bitstream.

For this, download from
https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html
the "AMD Unified Installer 2022.2 SFD"
somewhere you have enough storage.

Create a `/opt/xilinx` with the right access mode so you can work in
it.

Materialize the installer files with

```bash
tar zxvf .../Xilinx_Unified_2022.2_1014_8888.tar.gz
```

Since the graphics Java installer might not work on modern Linux
distributions like Ubuntu 22.04, use the batch-mode version:

```bash
Xilinx_Unified_2022.2_1014_8888/xsetup --xdebug --batch Install --location /opt/xilinx --agree XilinxEULA,3rdPartyEULA --product "Vitis" --edition "Vitis Unified Software Platform"
```

Note that if you want to uninstall this later, you can typically use:

```bash
/opt/xilinx/.xinstall/Vitis_2022.2/xsetup --batch Uninstall
```

If you have a Vitis framework with a version from before 2022, you
might be hit by the 2022+ year `int` overflow bug with an error
message like:

```
caught Tcl error: ERROR: '2202011410' is an invalid argument. Please specify an integer value.
```

To avoid this, follow the recipe from
https://support.xilinx.com/s/article/76960 after downloading the patch:

```bash
# The patching script assume working in the top AMD tool directory
cd /opt/xilinx
unzip ..../y2k22_patch-1.2.zip
python y2k22_patch/patch.py
```


## Boot on a specific kernel

If for some reason you need to boot a specific Linux kernel because
this is the only way to have XRT compiling and running with it, here
is a recipe.

Assuming for example you have an old kernel known to work, hold it to
make sure that it is not uninstalled during an upgrade with for
example:

```bash
sudo apt-mark hold linux-headers-5.11.0-41 linux-headers-5.11.0-41-generic \
    linux-image-5.11.0-41-generic linux-modules-5.11.0-41-generic \
    linux-modules-extra-5.11.0-41-generic linux-tools-5.11.0-41 \
    linux-tools-5.11.0-41-generic
```

> :memo: More generally it is useful to hold any old kernel just in case
> you discover later that it does not work anymore with a new kernel
> which was installed automatically...

Then you might want to boot by default with this kernel, for example
if you are not in front of the machine to select the right kernel with
the menu.

A semi-manual way can be to use instead of `GRUB_DEFAULT=0` in
`/etc/default/grub` to boot the first menu entry to have the
configuration:

```bash
#GRUB_DEFAULT=0
GRUB_SAVEDEFAULT=true
GRUB_DEFAULT=saved
```

so the default boot will be the previous one explicitly selected by
the menu.

A more explicit way to select a kernel is to figure out the menu entry
you are interested in by looking at `/boot/grub/grub.cfg` and update
`/etc/default/grub` with its reference, like:

```bash
#GRUB_DEFAULT=0
GRUB_DEFAULT='gnulinux-advanced-bedd64b9-a611-4ccf-ad12-8f703a5df1da>gnulinux-5.11.0-41-generic-advanced-bedd64b9-a611-4ccf-ad12-8f703a5df1da'
```

In any case, before rebooting, you need to update the boot
configuration from the modified `/etc/default/grub` by running:

```bash
sudo update-grub
```


## Installing the AMD XRT runtime

First be careful that `LD_LIBRARY_PATH` does not have the XRT
libraries in it because it might mess up with the unit tests linking
in the build process.

> :warning: there is currently a bug
> https://github.com/Xilinx/XRT/issues/6180 in hardware-emulation
> runtime where the simulator might not shutdown correctly.
> So you can try the branch from
> https://github.com/Xilinx/XRT/pull/6269

```bash
# Get the latest AMD runtime with its submodules
# Use either ssh
git clone --recurse-submodules git@github.com:Xilinx/XRT.git
# or https according to your usual method
git clone --recurse-submodules https://github.com/Xilinx/XRT.git
cd XRT/build
# If you need to clean the build from a previous compilation, try:
#  git clean -f -d -x .
# Install the required packages, which might fail because of liberal assumptions
sudo ../src/runtime_src/tools/scripts/xrtdeps.sh
# Setup the Vitis location so the Microblaze GCC can be used to
# compile the ERT XRT runtime
export XILINX_VITIS=/opt/xilinx/Vitis/2022.2
# If you want ccache to be used to amortize on several compilations
#  PATH="/usr/lib/ccache:$PATH"
# Compile the AMD runtime, use option to survive to some warning if any
# on some modern OS and compilers
./build.sh -disable-werror
# Install the runtime into /opt/xilinx/xrt and compile/install
# the Linux kernel drivers (adapt to the real name if different)
udo apt install --reinstall ./Release/xrt_202410.2.17.0_23.10-amd64-xrt.deb
sudo apt install --reinstall ./Release/xrt_202410.2.17.0_23.10-amd64-xbflash2.deb
```

It will install the user-mode XRT runtime and at least compile and
install the AMD device driver modules for the current running
kernel, even if it fails for the other kernels installed on the
machine. If you do not plan to run on a real FPGA board but only use
software or hardware emulation instead, it does not matter if the
kernel device driver is not compiled since it will not be used.

Note that if for some reasons you want to use a debug version of XRT,
use this recipe instead:

```bash
cd Debug
# You need to make explicitly the Debug package because it is not made
# by default
make package
# Install the runtime into /opt/xilinx/xrt and compile/install
# the Linux kernel drivers (adapt to the real name if different)
sudo apt install --reinstall ./xrt_202320.2.16.0_23.04-amd64-xrt.deb
sudo apt install --reinstall ./xrt_202320.2.16.0_23.04-amd64-xbflash2.deb
```

If you also want a debug and verbose version which traces all the
XRT calls, try compiling with `./dbg.sh -noccache`.

> :memo: The Linux kernel driver is actually compiled during the
> installation of the `.deb` package for the currently running
> kernel. If the compilation fails because of some incompatibilities,
> look at [section Boot on a specific kernel](#boot-specific-kernel)
> and reinstall the `.deb` after booting with another kernel.

> :memo: If the compilation fails for some of the Linux kernels of the
> system but works for the kernel you are interested in, it might
> prevent the kernel drivers to be loaded immediately. In that case
> you might want to avoid 1 reboot by loading them explicitly with:
>
> ```bash
> sudo modprobe --all xocl xclmgmt
> ```

Check that the FPGA board is detected:

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt examine
```

which should display something similar to:

```
System Configuration
  OS Name              : Linux
  Release              : 5.11.0-41-generic
  Version              : #45-Ubuntu SMP Fri Nov 5 11:37:01 UTC 2021
  Machine              : x86_64
  CPU Cores            : 20
  Memory               : 64235 MB
  Distribution         : Ubuntu 21.10
  GLIBC                : 2.34
  Model                : Precision Tower 5810

XRT
  Version              : 2.13.0
  Branch               : master
  Hash                 : 79d9180de2d15026ef7c50dda76a7df210d56c9e
  Hash Date            : 2021-12-14 17:10:50
  XOCL                 : unknown, unknown
  XCLMGMT              : 2.13.0, 79d9180de2d15026ef7c50dda76a7df210d56c9e

Devices present
  [0000:04:00.0] : xilinx_u200_GOLDEN_5  NOTE: Device not ready for use
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
example it is possible to use a `2018.3` target platform with Vitis `2022.2`,
and a version for Ubuntu `18.04` on a more recent version of Ubuntu or
Debian.

Install the target platforms according to documentation
https://docs.xilinx.com/r/en-US/ug1301-getting-started-guide-alveo-accelerator-cards
which is typically:

```bash
tar zxvf xilinx-u200_2022.1_2021_1021_1001-all.deb.tar.gz
# Install all of them at once to hide some dependency error (at least
# in 2021)
sudo apt install ./xilinx-sc-fw-u200-u250_4.6.18-1.ff327cc_all.deb \
 ./xilinx-cmc-u200-u250_1.2.20-3358356_all.deb \
 ./xilinx-u200-gen3x16-xdma-validate_1-3209073_all.deb \
 ./xilinx-u200-gen3x16-xdma-base_1-3209015_all.deb

sudo apt install ./xilinx-u200-gen3x16-xdma-1-202110-1-dev_1-3221508_all.deb
```

from where they have been downloaded or adapt the paths to them.

> :warning: Some packages have been shipped by AMD with a broken
> manifest, which might lead to some warning every time you use APT
> related package management commands, like:
>
> ```
> dpkg: warning: parsing file '/var/lib/dpkg/status' near line 138044 package 'xilinx-cmc-u200-u250':
>  missing 'Maintainer' field
> ```
>
> In that case edit `/var/lib/dpkg/status` around the indicated line
> and fix the line `Maintainer: Xilinx Inc.` by removing the leading
> spaces which were inserted by error.


### Flash and test the board

If you do not want to use a real board or if you have access to a
machine with a board which is already configured, like an XACC cluster
https://xilinx.github.io/xacc, skip this section.

If you want to install a real board, follow the recipe from
https://www.xilinx.com/support/documentation/boards_and_kits/accelerator-cards/1_9/ug1301-getting-started-guide-alveo-accelerator-cards.pdf
about how to correctly generate the exact flashing command.

> :warning: It looks like there are some issues to program an FPGA
> board when several versions of platforms are installed on the
> machine. If you need several platforms, uninstall the unused one
> when updating the FPGA board firmware and you can reinstall them
> afterwards.

Typically you run:

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt examine --report all --device 0000:04:00.0
System Configuration
  OS Name              : Linux
  Release              : 5.11.0-41-generic
  Version              : #45-Ubuntu SMP Fri Nov 5 11:37:01 UTC 2021
  Machine              : x86_64
  CPU Cores            : 20
  Memory               : 64235 MB
  Distribution         : Ubuntu 21.10
  GLIBC                : 2.34
  Model                : Precision Tower 5810

XRT
  Version              : 2.13.0
  Branch               : HEAD
  Hash                 : 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb
  Hash Date            : 2021-12-21 14:17:06
  XOCL                 : 2.13.0, 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb
  XCLMGMT              : 2.13.0, 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb

Devices present
  [0000:04:00.0] : xilinx_u200_GOLDEN_5  NOTE: Device not ready for use


----------------------------------------
1/1 [0000:04:00.0] : xilinx_u200_GOLDEN
----------------------------------------
Flash properties
  Type                 : spi
  Serial Number        : N/A

Device properties
  Type                 : u200
  Name                 : N/A
  Config Mode          : 3715565296
  Max Power            : N/A

Flashable partitions running on FPGA
  Platform             : xilinx_u200_GOLDEN_5
  SC Version           : INACTIVE
  Platform ID          : N/A

Flashable partitions installed in system
  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.6.18
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD

WARNING  : Device is not up-to-date.

Mechanical
  ERROR: Could not open device with index '0'

Firewall
  Level -- --: -- --

Mailbox
  ERROR: Failed to find subdirectory for mailbox under /sys/bus/pci/devices/0000:04:00.0


CMC
```

to get the information about the installed target platform and you
translate this into a flashing command according to the parameters
above or just follow the information you got when installing
previously the deployment target platform:

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt program --device 0000:04:00.0 --base --image xilinx_u200_gen3x16_xdma_base_1
----------------------------------------------------
Device : [0000:04:00.0]

Current Configuration
  Platform             : xilinx_u200_GOLDEN_5
  SC Version           : INACTIVE
  Platform ID          : N/A


Incoming Configuration
  Deployment File      : partition.xsabin
  Deployment Directory : /lib/firmware/xilinx/a2d4f3cf5b7a0b7b70f9da589cb5b3cd
  Size                 : 95,405,680 bytes
  Timestamp            : Tue Dec 21 14:39:07 2021

  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.6.18
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD
----------------------------------------------------
Actions to perform:
  [0000:04:00.0] : Program base (FLASH) image
  [0000:04:00.0] : Program Satellite Controller (SC) image
----------------------------------------------------
Are you sure you wish to proceed? [Y/n]: 

INFO: Satellite Controller (SC) images are the same.
[0000:04:00.0] : Updating base (e.g., shell) flash image...
Bitstream guard installed on flash @0x1002000
Extracting bitstream from MCS data:
.................................
Extracted 34437940 bytes from bitstream @0x1002000
Writing bitstream to flash 0:
.................................
Bitstream guard removed from flash
INFO     : Base flash image has been programmed successfully. 
----------------------------------------------------
Report
  [0000:04:00.0] : Satellite Controller (SC) is either up-to-date, fixed, or not installed. No actions taken.
  [0000:04:00.0] : Successfully flashed the base (e.g., shell) image

Device flashed successfully.
****************************************************
Cold reboot machine to load the new image on device.
****************************************************
```

Unfortunately you need to "cold reboot" the machine to have the new
target platform loaded inside the FPGA, which means to really
power-off the machine so the new instantiated PCIe interface of the
card can actually be rediscovered by the host machine (everything is
configurable with an FPGA!).

After rebooting you can look at the FPGA configuration:

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt examine --report all --device 0000:04:00.0
System Configuration
  OS Name              : Linux
  Release              : 5.11.0-41-generic
  Version              : #45-Ubuntu SMP Fri Nov 5 11:37:01 UTC 2021
  Machine              : x86_64
  CPU Cores            : 20
  Memory               : 64235 MB
  Distribution         : Ubuntu 21.10
  GLIBC                : 2.34
  Model                : Precision Tower 5810

XRT
  Version              : 2.13.0
  Branch               : HEAD
  Hash                 : 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb
  Hash Date            : 2021-12-21 14:17:06
  XOCL                 : 2.13.0, 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb
  XCLMGMT              : 2.13.0, 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb

Devices present
  [0000:04:00.0] : xilinx_u200_gen3x16_xdma_base_1 mgmt(inst=1024) 


-----------------------------------------------------
1/1 [0000:04:00.0] : xilinx_u200_gen3x16_xdma_base_1
-----------------------------------------------------
Flash properties
  Type                 : spi
  Serial Number        : 21290605K03Y

Device properties
  Type                 : u200
  Name                 : ALVEO U200 PQ
  Config Mode          : 7
  Max Power            : 225W

Flashable partitions running on FPGA
  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.2
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD
  Interface UUID       : 15FB8DA1-F552-A9F9-23DE-6DC54AA8968F

Flashable partitions installed in system
  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.6.18
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD


  Mac Address          : 00:0A:35:06:9F:8A
                       : 00:0A:35:06:9F:8B

WARNING  : SC image on the device is not up-to-date.

Mechanical
  Fans
    FPGA Fan 1
      Critical Trigger Temp : 38 C
      Speed                 : 1107 RPM

Firewall
  Level 0 : 0x0 (GOOD)

Mailbox
  Total bytes received   : 704 Bytes
  Unknown                : 0 
  Test msg ready         : 0 
  Test msg fetch         : 0 
  Lock bitstream         : 0 
  Unlock bitstream       : 0 
  Hot reset              : 0 
  Firewall trip          : 0 
  Download xclbin kaddr  : 0 
  Download xclbin        : 0 
  Reclock                : 0 
  Peer data read         : 5 
  User probe             : 1 
  Mgmt state             : 0 
  Change shell           : 0 
  Reprogram shell        : 0 
  P2P bar addr           : 0 

CMC
  Status : 0x1390001 (SINGLE_SENSOR_UPDATE_ERR)
  err time : 1640129064 sec

  Runtime clock scaling feature :
    Supported : false
    Enabled : false
    Critical threshold (clock shutdown) limits:
      Power : 0 W
      Temperature : 0 C
    Throttling threshold limits:
      Power : 0 W
      Temperature : 0 C
    Power threshold override:
      Override : false
      Override limit : 0 W
    Temperature threshold override:
      Override : false
      Override limit : 0 C
```

In that case the `WARNING  : SC image on the device is not
up-to-date.` shows that there is still need to program it by
repeating the same program command:

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt program --device 0000:04:00.0 --base --image xilinx_u200_gen3x16_xdma_base_1
----------------------------------------------------
Device : [0000:04:00.0]

Current Configuration
  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.2
  Platform ID          : 0xa2d4f3cf5b7a0b7b


Incoming Configuration
  Deployment File      : partition.xsabin
  Deployment Directory : /lib/firmware/xilinx/a2d4f3cf5b7a0b7b70f9da589cb5b3cd
  Size                 : 95,405,680 bytes
  Timestamp            : Tue Dec 21 14:39:07 2021

  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.6.18
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD
----------------------------------------------------
Actions to perform:
  [0000:04:00.0] : Program Satellite Controller (SC) image
----------------------------------------------------
Are you sure you wish to proceed? [Y/n]: 

[0000:04:00.0] : Updating Satellite Controller (SC) firmware flash image...
Stopping user function...
INFO     : found 5 sections
[PASSED] : SC successfully updated < 37s >
INFO     : Loading new firmware on SC
.......

INFO: Base (e.g., shell) flash images are the same.
----------------------------------------------------
Report
  [0000:04:00.0] : Successfully flashed the Satellite Controller (SC) image
  [0000:04:00.0] : Base (e.g., shell) image is up-to-date.  No actions taken.

Device flashed successfully.
******************************************************************
Warm reboot is required to recognize new SC image on the device.
******************************************************************
```

Good news here, only a normal reboot is required now.

You can check with:

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt examine --report all --device 0000:04:00.0
System Configuration
  OS Name              : Linux
  Release              : 5.11.0-41-generic
  Version              : #45-Ubuntu SMP Fri Nov 5 11:37:01 UTC 2021
  Machine              : x86_64
  CPU Cores            : 20
  Memory               : 64235 MB
  Distribution         : Ubuntu 21.10
  GLIBC                : 2.34
  Model                : Precision Tower 5810

XRT
  Version              : 2.13.0
  Branch               : HEAD
  Hash                 : 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb
  Hash Date            : 2021-12-21 14:17:06
  XOCL                 : 2.13.0, 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb
  XCLMGMT              : 2.13.0, 37f368eae34cb70b8c3dc1c840d02e59b2b6b8cb

Devices present
  [0000:04:00.0] : xilinx_u200_gen3x16_xdma_base_1 mgmt(inst=1024) 


-----------------------------------------------------
1/1 [0000:04:00.0] : xilinx_u200_gen3x16_xdma_base_1
-----------------------------------------------------
Flash properties
  Type                 : spi
  Serial Number        : 21290605K03Y

Device properties
  Type                 : u200
  Name                 : ALVEO U200 PQ
  Config Mode          : 7
  Max Power            : 225W

Flashable partitions running on FPGA
  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.6.18
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD
  Interface UUID       : 15FB8DA1-F552-A9F9-23DE-6DC54AA8968F

Flashable partitions installed in system
  Platform             : xilinx_u200_gen3x16_xdma_base_1
  SC Version           : 4.6.18
  Platform UUID        : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD


  Mac Address          : 00:0A:35:06:9F:8A
                       : 00:0A:35:06:9F:8B
                       : 00:0A:35:06:9F:8C
                       : 00:0A:35:06:9F:8D


Mechanical
  Fans
    FPGA Fan 1
      Critical Trigger Temp : 41 C
      Speed                 : 1108 RPM

Firewall
  Level 0 : 0x0 (GOOD)

Mailbox
  Total bytes received   : 576 Bytes
  Unknown                : 0 
  Test msg ready         : 0 
  Test msg fetch         : 0 
  Lock bitstream         : 0 
  Unlock bitstream       : 0 
  Hot reset              : 0 
  Firewall trip          : 0 
  Download xclbin kaddr  : 0 
  Download xclbin        : 0 
  Reclock                : 0 
  Peer data read         : 4 
  User probe             : 1 
  Mgmt state             : 0 
  Change shell           : 0 
  Reprogram shell        : 0 
  P2P bar addr           : 0 

CMC
  Status : 0x0 (GOOD)
  Runtime clock scaling feature :
    Supported : false
    Enabled : false
    Critical threshold (clock shutdown) limits:
      Power : 0 W
      Temperature : 0 C
    Throttling threshold limits:
      Power : 0 W
      Temperature : 0 C
    Power threshold override:
      Override : false
      Override limit : 0 W
    Temperature threshold override:
      Override : false
      Override limit : 0 C
```

Then after rebooting, you can check with a pre-compiled FPGA program
provided by the target platform that the board is working correctly
with (the device id below is to adapt to your card):

```bash
sudo /opt/xilinx/xrt/bin/xbutil validate --verbose --device 0000:04:00.1
Verbose: Enabling Verbosity
Starting validation for 1 devices

Validate Device           : [0000:04:00.1]
    Platform              : xilinx_u200_gen3x16_xdma_base_1
    SC Version            : 4.6.18
    Platform ID           : A2D4F3CF-5B7A-0B7B-70F9-DA589CB5B3CD
-------------------------------------------------------------------------------
Test 1 [0000:04:00.1]     : Aux connection 
    Description           : Check if auxiliary power is connected
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 2 [0000:04:00.1]     : PCIE link 
    Description           : Check if PCIE link is active
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 3 [0000:04:00.1]     : SC version 
    Description           : Check if SC firmware is up-to-date
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 4 [0000:04:00.1]     : Verify kernel 
    Description           : Run 'Hello World' kernel test
    Xclbin                : /opt/xilinx/firmware/u200/gen3x16-xdma/base/test/verify.xclbin
    Testcase              : /opt/xilinx/xrt/test/validate.exe
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 5 [0000:04:00.1]     : DMA 
    Description           : Run dma test
    Details               : Host -> PCIe -> FPGA write bandwidth = 8950.9 MB/s
                            Host <- PCIe <- FPGA read bandwidth = 11743.5 MB/s
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 6 [0000:04:00.1]     : iops 
    Description           : Run scheduler performance measure test
    Xclbin                : /opt/xilinx/firmware/u200/gen3x16-xdma/base/test/verify.xclbin
    Testcase              : /opt/xilinx/xrt/test/xcl_iops_test.exe
    Details               : IOPS: 469984 (verify)
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 7 [0000:04:00.1]     : Bandwidth kernel 
    Description           : Run 'bandwidth kernel' and check the throughput
    Xclbin                : /opt/xilinx/firmware/u200/gen3x16-xdma/base/test/bandwidth.xclbin
    Testcase              : /opt/xilinx/xrt/test/kernel_bw.exe
    Details               : Throughput (Type: DDR) (Bank count: 4) : 67197.8MB/s
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 8 [0000:04:00.1]     : Peer to peer bar 

    Description           : Run P2P test
    Details               : P2P bar is not enabled
    Test Status           : [SKIPPED]
-------------------------------------------------------------------------------
Test 9 [0000:04:00.1]     : Memory to memory DMA 

    Description           : Run M2M test
    Details               : M2M is not available
    Test Status           : [SKIPPED]
-------------------------------------------------------------------------------
Test 10 [0000:04:00.1]    : Host memory bandwidth test 

    Description           : Run 'bandwidth kernel' when host memory is enabled
    Details               : Host memory is not enabled
    Test Status           : [SKIPPED]
-------------------------------------------------------------------------------
Test 11 [0000:04:00.1]    : vcu 
    Description           : Run decoder test
    Details               : Verify xclbin not available or shell partition is not
                            programmed. Skipping validation.
    Test Status           : [SKIPPED]
-------------------------------------------------------------------------------
Validation completed
```


## Compile the SYCL compiler

Some packages are required to use the compiler. It can be done for
example on Debian/Ubuntu with:
```bash
sudo apt update
sudo apt install python3-posix-ipc
```

Building SYCL can be done with Python scripts:

```bash
# Pick some place where SYCL has to be compiled, such as:
DPCPP_HOME=~/sycl_workspace
mkdir $DPCPP_HOME
cd $DPCPP_HOME
# You can also try --branch sycl/unified/next for a bleeding edge experience
git clone --branch sycl/unified/master git@github.com:triSYCL/sycl.git llvm
# Use --xrt is to enable the optional XRT plugin. This is a replacement for the OpenCL plugin
# because XRT offers more control and expressiveness on the hardware
python3 $DPCPP_HOME/llvm/buildbot/configure.py --xrt
python3 $DPCPP_HOME/llvm/buildbot/compile.py
```

These scripts have many options which can be displayed when using the
`--help` option. For example to configure with CUDA and HIP support,
shared libraries and producing a compiler database to be
used by tools like LSP server like `clangd`:

```bash
python3 $DPCPP_HOME/llvm/buildbot/configure.py --xrt --cuda --hip \
  --shared-libs --cmake-opt="-DCMAKE_EXPORT_COMPILE_COMMANDS=1"
```
Use `--build-type=Debug` for a debug build.

For more control, see [section Build](#build).

# Usage
## Compiling and running a SYCL application

The typical environment is setup with something like

```bash
# The place where SYCL has been compiled (to update to the real location):
export DPCPP_HOME=~/sycl_workspace
# Most of the export here are optional but useful when using in a
# subshell or through tmux.
# The version of Vitis you want to use
export XILINX_VERSION=2022.2
# The target platform for the FPGA board model
export XILINX_PLATFORM=xilinx_u200_gen3x16_xdma_1_202110_1
# Where all the AMD tools are
export XILINX_ROOT=/opt/xilinx
# Where the SYCL compiler binaries are:
export SYCL_BIN_DIR=$DPCPP_HOME/llvm/build/bin
export XILINX_XRT=$XILINX_ROOT/xrt
export XILINX_VITIS=$XILINX_ROOT/Vitis/$XILINX_VERSION
export XILINX_HLS=$XILINX_ROOT/Vitis_HLS/$XILINX_VERSION
export XILINX_VIVADO=$XILINX_ROOT/Vivado/$XILINX_VERSION
# Add the various tools in the PATH
PATH=$PATH:$SYCL_BIN_DIR:$XILINX_XRT/bin:$XILINX_VITIS/bin:$XILINX_HLS/bin:$XILINX_VIVADO/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XILINX_XRT/lib:$XILINX_VITIS/lib/lnx64.o:$XILINX_VITIS_HLS/lib/lnx64.o:$DPCPP_HOME/llvm/build/lib
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
  open source AMD HLS front-end. This way, anything supported by
  AMD HLS C++ should be supported at some point in the future;

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

Architecture provided to the `--sycl-targets` Clang option selects the
compilation mode. Supported architectures are:

|                                    | Software simulation | Hardware emulation  | Hardware        |
|------------------------------------|---------------------|---------------------|-----------------|
| SPIR compilation flow (deprecated) | `fpga64_sw_emu`     | `fpga64_hw_emu`     | `fpga64_hw`     |
| HLS compilation flow               | Unsupported yet     | `fpga64_hls_hw_emu` | `fpga64_hls_hw` |

Only one `fpga64_*` architecture is allowed in the `--sycl-targets`
option.

The SYCL HLS compilation flow does not support software emulation because
of internal AMD issue https://jira.xilinx.com/browse/CR-1099885
But as SYCL allows also execution on a CPU device, it can replace the
backend software emulation.


### Picking the right device

The SYCL framework displays the available SYCL devices by running the
`sycl-ls` command, with an optional `--verbose` parameter for more
information.

The XRT runtime exposes the real FPGA available on the machine or some
emulated FPGA according to the `XCL_EMULATION_MODE` environment
variable: which can take 3 values according to the required emulation:
- `hw_emu`, for hardware emulation where an RTL simulator is used to
  run (slowly) the kernels instead of using a real FPGA and is the
  only option that matters here;
- `hw`, for real hardware execution, equivalent to not defining
  `XCL_EMULATION_MODE`;
- `sw_emu`, for software emulation, which is not very useful in SYCL
  since the code is already executable on CPU as plain single-source
  C++, but makes sense for HLS C/C++ or OpenCL.

When running a SYCL program on AMD FPGA, the runtime takes care
of this variable according to the compilation option, so it is not
necessary to set it.

But to list the available SYCL devices on the machine, setting the
variable to have a list of available devices according whether you
want to see simulated devices or not.

For example, on a machine without a physical FPGA, running the
`sycl-ls` command can show:
```bash
>- sycl-ls
XRT build version: 2.14.0
Build hash: 3d71c4e867bf91e789f89499a97b95708331d7e7
Build date: 2022-04-12 13:48:07
Git branch: master
PID: 421146
UID: 1000
[Tue Apr 26 01:04:14 2022 GMT]
HOST: rk-xsj
EXE: /home/rkeryell/Xilinx/Projects/LLVM/worktrees/xilinx/llvm/build/bin/sycl-ls
[XRT] ERROR: No devices found
[opencl:cpu:0] Intel(R) CPU Runtime for OpenCL(TM) Applications, Intel(R) Xeon(R) E-2176M  CPU @ 2.70GHz 2.1 [18.1.0.0920]
[opencl:cpu:1] Portable Computing Language, pthread-Intel(R) Xeon(R) E-2176M  CPU @ 2.70GHz 1.2 [1.8]
[host:host:0] SYCL host platform, SYCL host device 1.2 [1.2]
```
with an error message ending with `[XRT] ERROR: No devices found`.

But by asking some hardware emulated devices, the installed platforms
show up:
```bash
>- XCL_EMULATION_MODE=hw_emu sycl-ls
[opencl:acc:0] Xilinx, xilinx_u200_gen3x16_xdma_1_202110_1 1.0 [1.0]
[opencl:cpu:1] Intel(R) CPU Runtime for OpenCL(TM) Applications, Intel(R) Xeon(R) E-2176M  CPU @ 2.70GHz 2.1 [18.1.0.0920]
[opencl:cpu:2] Portable Computing Language, pthread-Intel(R) Xeon(R) E-2176M  CPU @ 2.70GHz 1.2 [1.8]
[host:host:0] SYCL host platform, SYCL host device 1.2 [1.2]
```
as `opencl:acc:0` which can be used to select the right device.

Note that the emulated AMD FPGA and the real AMD FPGA might have
different names in the OpenCL or XRT backends. For example our Alveo
U200 appears as `xilinx_u200_gen3x16_xdma_1_202110_1` in emulation but
`xilinx_u200_gen3x16_xdma_base_1` in hardware. This has to be taken
into account when using explicit device selection by name.

There is bug in XRT when an AMD FPGA device is used from both the
OpenCL and XRT platforms https://github.com/Xilinx/XRT/issues/7226
which results in a segmentation violation. Since XRT is the preferred
approach nowadays, if you do not use OpenCL, just temporary rename the
AMD FPGA ICD file with
```
sudo mv /etc/OpenCL/vendors/xilinx.icd /etc/OpenCL/vendors/xilinx.icd.bak
```
to make the FPGA no longer visible from OpenCL. By keeping the file as
a backup, you can always rename it back when you need to use the
OpenCL platform for the AMD FPGA.

The examples provided here often rely on the SYCL default selector
whose behavior can be influenced by the
[`ONEAPI_DEVICE_SELECTOR`](EnvironmentVariables.md#oneapi_device_selector)
environment variable, among others. Thus, to run an example on the
AMD FPGA shown by the previous `sycl-ls`, the environment
variable can be set with:
```bash
>- export ONEAPI_DEVICE_SELECTOR=xrt:fpga
```
Beware that setting this variable also change the view from `sycl-ls` itself.


### Small examples

See section [Picking the right device](#picking-the-right-device) to
set correctly the `ONEAPI_DEVICE_SELECTOR` environment variable to select
the right AMD FPGA device first.

To run an example from the provided examples

- with hardware emulation:

  ```bash
  cd $DPCPP_HOME/llvm/sycl/test/vitis/simple_tests
  # Instruct the compiler and runtime to use FPGA hardware emulation with HLS flow
  # Compile the SYCL program down to a host fat binary including the RTL for simulation
  $SYCL_BIN_DIR/clang++ -std=c++20 -fsycl -fsycl-targets=fpga64_hls_hw_emu \
    single_task_vector_add.cpp -o single_task_vector_add
  # Run the hardware emulation
  ./single_task_vector_add
  ```

- with real hardware execution on FPGA:

  ```bash
  cd $DPCPP_HOME/llvm/sycl/test/vitis/simple_tests
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
files... including AMD `.xpr` projects which can be used by Vivado
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

- Run the `vitis` test suite on CPU just to check the basic API
  (around 1 minute):

  ```bash
  cd $DPCPP_HOME/llvm/build
  # Running tests with the CPU backend
  cmake --build . --parallel `nproc` --target check-sycl-vitis-cpu
  ```

- Run the `vitis` test suite with hardware emulation (HLS flow):

  ```bash
  cd $DPCPP_HOME/llvm/build
  # Running tests with the OpenCL backend
  cmake --build . --parallel `nproc` --target check-sycl-vitis-opencl
  # Running tests with the XRT backend
  cmake --build . --parallel `nproc` --target check-sycl-vitis-xrt
  ```

This takes usually 45-60 minutes with a good CPU.

- Run the `vitis` test suite with real hardware execution on FPGA (HLS flow):

  ```bash
  cd $DPCPP_HOME/llvm/build
  # Running tests with the OpenCL backend
  cmake --build . --parallel `nproc` --target check-sycl-vitis-opencl-hw
  # Running tests with the XRT backend
  cmake --build . --parallel `nproc` --target check-sycl-vitis-xrt-hw
  ```

This takes usually 10+ hours.

### Running a bigger example on real FPGA

To run a SYCL translation of
https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/edge_detection

```bash
cd $DPCPP_HOME/llvm/sycl/test/vitis/edge_detection
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

## AMD FPGA extension

In order to give user more control over how is implemented the kernel on hardware,
a few SYCL extensions are provided as a replacement of Vitis HLS `HLS` pragmas.

These extensions requires the inclusion of the `sycl/ext/xilinx/fpga.hpp`
header, and requires compiling with support of C++20 or more.

### Pipelining

Pipelining support comes from the `sycl/ext/xilinx/fpga/pipeline.hpp` header.

A few pipeline types are defined, that are used to parametrize pipelining decorators.

The `pipeline_style` enum describes the different pipeline style supported by HLS:

+ `stall` pipelines only run when input data is present and stops otherwise (default),
+ `flushable` pipelines run when input data is present or if there is still data in the pipeline,
+ `free_running` pipelines run without stalling (simpler logic but more energy consumption).

Two types are also defined to describe more specifically loop pipeline behavior:

+ `no_rewind_pipeline` loop pipelines are flushed between two execution of the full loop, while
+ `rewind_pipeline` are not.

Finally, a few types are present to constrain the pipeline initialization interval :

+ `auto_ii` let Vitis determine the pipeline II,
+ `constrained_ii<Value>` asks Vitis to create a pipeline with an II of exactly Value cycles,
+ `disable_pipeline` force the absence of pipelining.

Two decorations exists to pipeline part of the code.

+ `sycl::ext::xilinx::pipeline_kernel<IIType, PipelineType>` is used to pipeline kernel,
+ `sycl::ext::xilinx::pipeline<IIType, RewindType, PipelineType>` is used to pipeline a loop.

Kernel pipelining:

```cpp
cgh.single_task(pipeline_kernel<constrained_ii<4>>(
  [=]{
    for (size_t i = 0 ; i < 42 ; ++i>) {
      OutAcc[i] = InAcc[i] * i;
    }
  }
));
```

Loop pipelining:

```cpp
chh.single_task([=]{
  // Pipeline the following loop
  for (size_t i = 0 ; i < 42 ; ++i>) {
    pipeline([&]{
      OutAcc[i] = InAcc[i] * i;
    });
  }
});
```

The lambda given to `pipeline` is executed for each step of
the loop, and the enclosing loop is pipelined.

Helpers to deactivate the pipeline `for` loop and kernel,
respectively `no_pipeline` and `unpipeline_kernel` are used in
a similar way.

### Dataflow decorators

These decorators are the equivalent of the
[HLS dataflow pragma](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-dataflow).

A kernel decorator `sycl::ext::xilinx::dataflow_kernel()` and a loop
decorator `sycl::ext::xilinx::dataflow()` exists, which are used in
a similar fashion to the equivalent pipeline annotation.

### Loop unrolling

The `sycl::ext::xilinx::unroll<UnrollType>()` decorator should be applied
to a lambda inside a `for` loop. The lambda is called at each loop
iteration and the enclosing loop is unrolled following UnrollType
semantics.

`UnrollType` can be one of the following:

+ `no_unrolling`: forbids unrolling of the enclosing loop,
+ `full_unrolling`: The loop will be replaced by expliciting all the iteration,
+ `checked_fixed_unrolling<unroll_factor>`: The iteration will be grouped
   by unroll_factor. If the total number of iteration is unknown or not
   n integral multiple of unroll_factor, extra hardware is generated
   to ensure the loop does not perform too much iterations.
+ `unchecked_fixed_unrolling<unroll_factor>`: Like `checked_fixed_unrolling`,
  but does not generate the extra hardware to ensure the loop exits at
  the correct iteration if the total number of iteration is not an
  integral multiple of unroll_factor.
  When the total number of iteration is only known at runtime, the user
  takes the responsibility that it will always be an integral
  multiple of unroll_factor.
  If it is known at compile time and does not verify the property,
  the backend compilation will fail.

### Pinning allocators to specific memory banks

The memory bank on which a buffer is copied on the device is
controllable via an accessor property.

As of now, buffer pinning is an all-or-nothing feature for kernel: either all 
buffer accessors should have a specific memory bank assignment, or none of them.

Two kinds of memory banks are supported: DDR and HBM.

The related accessor property is `sycl::ext::xilinx::ddr_bank<BANK_ID>` for the
former and `sycl::ext::xilinx::hbm_bank<BANK_ID>` for the later.

Example:

```cpp
sycl::queue queue;
sycl::buffer<int> a_buff, b_buff, c_buff;

/*
Fill a_buff and b_buff
*/

queue.submit([&](sycl::handler &cgh) {
  sycl::ext::oneapi::accessor_property_list property_bank_1{sycl::ext::xilinx::ddr_bank<1>};
  sycl::ext::oneapi::accessor_property_list property_bank_2{sycl::ext::xilinx::ddr_bank<2>};
  sycl::ext::oneapi::accessor_property_list property_bank_3{sycl::ext::xilinx::ddr_bank<3>};

  const sycl::accessor a_acc{a_buff, cgh, sycl::read_only, property_bank_1};
  const sycl::accessor b_acc{b_buff, cgh, sycl::read_only, property_bank_2};
  sycl::accessor c_acc{c_buff, cgh, sycl::write_only, property_bank_3};

  cgh.single_task<class VectorAddDDR>([=] {
    for (std::size_t i = 0 ; i < N ; ++i) {
      c_acc[i] = a_acc[i] + b_acc[i];
    }
  });
});
}

```

In this example, `a_buff` materialization on device will be stored on DDR
bank 1, `b_buff` materialization on DDR bank 2 and `c_buff` materialization on
DDR bank 3.

**As of now, having accessor on the same buffer with different memory location is not supported.**

### Array partitioning

Partitioning an array allows having faster accesses to that array while using more hardware.
all code will be shown with
```cpp
using namespace sycl::ext::xilinx
```

`partition_ndarray<int, sycl::ext::xilinx::dim<2, 3>, partition::complete<>>` is equivalent to the C-style array `int array[2][3]` but we specified a complete partitioning like the [HLS array_partition pragma](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition).

+ `partition::complete<>` each element will be in an individual register
+ `partition::block<4>` the array will be partitioned into `4` continuous blocks.
+ `partition::cyclic<6>` the array will be partitioned into `6` interleaving sections.

`partition_array` is an alias for 1d arrays.

Example:
```cpp
sycl::queue queue;
sycl::buffer<int> a_buff, b_buff;

/// Fill the buffers

queue.submit([&](sycl::handler &cgh) {
  const sycl::accessor a_acc{a_buff, cgh, sycl::read_only};
  sycl::accessor b_acc{b_buff, cgh, sycl::write_only};
  cgh.single_task<class Calculation>([=] {
    partition_ndarray<int, sycl::ext::xilinx::dim<2, 3>, partition::complete<>> array = {{1, 2, -1}, {3, 4, -2}};
    for (std::size_t i = 0 ; i < N ; ++i) {
      c_acc[i] = b_acc[i] * array[i % 2][i % 3];
    }
  });
});
```

for more examples see [this test case](../test/vitis/simple_tests/partion_ndarray.cpp) or [this one](../test/vitis/edge_detection/edge_detection.cpp)

### Multiple annotations

It is also possible to apply multiple annotations on the same loop by using ``sycl::ext::xilinx::annot`` like the following:

```cpp
/// assuming
using namespace sycl::ext::xilinx

for (int i = 0; i < a.get_size(); i++)
  annot<pipeline<>, unroll<checked_fixed_unrolling<8>>>([&] { a[i] = 0; });
```

This loop will be pipelined with default settings and unrolled by a factor of 8

## AMD Macros

``__SYCL_XILINX_SW_EMU_MODE__`` will be defined when compiling device code in sw_emu mode

``__SYCL_XILINX_HW_EMU_MODE__`` will be defined when compiling device code in hw_emu mode

``__SYCL_XILINX_HW_MODE__`` will be defined when compiling device code in hw mode

when compiling host code none of the ``__SYCL_XILINX_*_MODE__`` macros will be defined.

``__SYCL_HAS_XILINX_DEVICE__`` will be defined on the host if one of
the specified targets is an AMD device or on an AMD device

## Known issues

### Xsim

When a SYCL application running in hw_emu is not properly terminated like killed from a debugger.
It is possible that the xsim process used for simulation will still be running. This process has some memory leaks so it is an issue.
[cleanup_xsimk.cpp](../tools/cleanup_xsimk.cpp) contains the sources for a process that will cleanup the xsims that are not in use every 5 minutes

### Shared Libs

XRT requires the XCL_EMULATION_MODE environnement to be:
 - unset for hw
 - set to hw_emu for hw_emu
 - set to sw_emu for sw_emu.

This is usually done automatically by the SYCL runtime while loading kernels when global constructors are running.
But when using dynamic libraries the global constructors of the libraries may happen after XRT is already loaded.
Causing XRT to fail to load binaries because it is loaded in the incorrect mode.
This issue can be bypassed by using XCL_EMULATION_MODE manually.

for example:
```cpp
/// the main.cpp
/// clang++ -g -fsycl -fsycl-targets=nvptx64-nvidia-cuda main.cpp
int main() {
  sycl::buffer<int> v{10};
  sycl::queue(sycl::gpu_selector_v).submit([&](auto &h) {
    auto a = sycl::accessor{v, h};
    h.parallel_for(a.size(), [=](int i) { a[i] = i; });
  });

  ((void (*)(sycl::buffer<int> &))dlsym(dlopen("my_sycl.so", RTLD_LAZY),"run"))(v);
}

/// lib.cpp
/// clang++ -g -fsycl -fsycl-targets=fpga64_hls_hw_emu -std=c++20 lib.cpp -fPIC -shared -o my_sycl.so
extern "C" void run(sycl::buffer<int> &v) {
  sycl::queue(xrt_selector_v).submit([&](auto &h) {
    auto a = sycl::accessor{v, h};
    h.parallel_for(a.size(), [=](int i) { a[i] *= a[i]; });
  });
}
```

# Implementation

## AMD FPGA SYCL compiler architecture

[Architecture of the AMD SYCL
compiler](design/AMD_FPGA_SYCL_compiler_architecture.rst) describes the
compiler architecture.

This document aims to cover the key differences of compiling SYCL for AMD
FPGAs. Things like building the compiler and library remain the same but other
things like the compiler invocation for AMD FPGA compilation is a little
different. As a general rule of thumb we're trying to keep things as close as we
can to the Intel implementation, but in some areas were still working on that.

One of the significant differences of compilation for AMD FPGAs
over the ordinary compiler directive is that AMD devices
require offline compilation of SYCL kernels to binary before being
wrapped into the end fat binary. The offline compilation of these
kernels is done by AMD's `v++` compiler rather than the SYCL
device compiler itself in this case. The device compiler's job is to
compile SYCL kernels to a format edible by `v++`, then take the output
of `v++` and wrap it into the fat binary as normal.

The current Intel SYCL implementation revolves around SPIR-V while
AMD's `v++` compiler can only ingest plain LLVM IR 6.x or LLVM IR
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


### Extra Notes

* The Driver ToolChain, currently makes some assumptions about the
  `Vitis` installation. For example, it assumes that `v++` is inside
  Vitis's `bin` folder and that the `lib` folder containing `SPIR`
  builtins that kernels are linked against are in a `lnx64/lib`
  directory relative to the `bin` folder parent.

## Debugging the SYCL implementation

### Debugging the driver and intermediate steps

During the compilation, a temporary directory is created in which all Vitis inputs, commands, outputs and logs are stored.
By default, this directory is deleted as soon as the compilation ends (even when it fails).

In order to keep it, set the `SYCL_VXX_KEEP_CLUTTER` environment variable to True.

The compiler will output a similar to

```
Temporary clutter in /tmp/EXECNAME-e5ece1pxk5rz43 will not be deleted
```

Informing you of where those files are kept (`/tmp/EXECNAME-e5ece1pxk5rz43` in this case).

### Debugging the SYCL runtime

There is 2 supported backends targeting AMD FPGA the XRT backend and the OpenCL backend.
if you are using the default device selector the xrt backend selected by:
```bash
export ONEAPI_DEVICE_SELECTOR=xrt:*
```
and the OpenCL backend can be selected by:
```bash
export ONEAPI_DEVICE_SELECTOR=opencl:*
```

Testing if it is possible to reproduce a bug on the other backend can give you more information about the bug.

Also the XRT backend support generating reproducer for debugging purposes (or bug reports). To generate a reproducer:

Run the program with SYCL_PI_XRT_REPRODUCER_PATH=path/to/reprod.out.cpp

Create a file named `reprod.cpp` with:
```cpp
#include <xrt/xrt_kernel.h>
#include <xrt.h>
#include <array>
#include <cassert>
#include <cstring>

int main() {

  /// insert the code here

  /// Edit the code below to validate the data
  /// int* a_c = name#.data();
  for (int i = 0; i < /*size*/; i++) {
    int res = i + i + 1;
    int val = a_c[i];
    assert(val == res);
  }
  printf("PASS\n");
}
```
copy the content of `path/to/reprod.out.cpp` into `reprod.cpp` where the comment says so.

The reproducer is going to leave comments with a TODO everywhere it couldn't automatically generate the code.
So find all the TODOs in the file. If your SYCL code is simple, there should be only 1 TODO which should be replaced by the path to the `xclbin` see the previous section to get the path to the `xclbin`.

Also edit the end of the `main` to adapt or remove the validation of data

then you can compile your reproducer via:
```bash
g++ -o reprod reprod.cpp -g -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lOpenCL -luuid -lxrt_coreutil
```

### Environnement variables

Some environment variables are very useful for debugging:

```bash
# Redirect the directory used as temporary directory
# for the compiler and various tools
export TMP=$DPCPP_HOME/../tmp

# SYCL_PI_TRACE should always be at least at 1, this make the SYCL runtime emit logs about which device is selected
export SYCL_PI_TRACE=1

# SYCL_PI_TRACE can be set at -1 to have full debug information but this is quite noisy
export SYCL_PI_TRACE=-1
```


### Clang flags

Some useful Clang flags:
* `-ccc-print-phases` outputs the graph of compilation phases;
* `-ccc-print-bindings` outputs the mapping from compilation phases to commands;
* `-###` outputs the commands to run to compile. The driver will create files to reserve them for those commands.

### Running a single test

To run a test from the test suite in isolation, use:

```bash
# The SYCL_TRIPLE variable can be changed to select hw or hw_emu, and the SYCL_PLUGIN can be changed to select opencl or xrt
/path/to/build/dir/bin/llvm-lit -v --param VITIS=only --param SYCL_TRIPLE=fpga64_hls_hw_emu-xilinx-linux --param SYCL_PLUGIN=xrt path/to/test.cpp
```

where all tests utilities must have been build for this to work.


### v++ Logs

The kinds of following errors are typical of a backend issue:

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


### llvm-reduce

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
./build-Release/bin/opt -verify -S -vxxIRDowngrader $1 -o $1.tmp.ll || exit 1

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
