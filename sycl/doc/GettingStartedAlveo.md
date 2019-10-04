Getting started with SYCL with a Xilinx FPGA U200 Alveo board and Ubuntu 19.04
==============================================================================

Disclaimer: nothing here is supported and this is all about a research
project.

We assume that you have the latest Ubuntu 19.04 version installed on
an `x86_64` machine.

It might work for some other version of Ubuntu or Debian with some
adaptations.


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


## Installing the Xilinx runtime

```bash
# Get the Xilinx runtime. You might try the master branch instead...
git clone --branch 2019.1 git@github.com:Xilinx/XRT.git
cd XRT/build
# Install the required packages
../src/runtime_src/tools/scripts/xrtdeps.sh
# Compile the Xilinx runtime
./build.sh
# Install the runtime and compile/install the Linux kernel driver
# (adapt to the real name if different)
sudo apt install --reinstall ./Release/xrt_201910.2.2.0_19.04-xrt.deb
```

It will install the user-mode XRT runtime and at least compile and
install the Xilinx device driver modules for the current running kernel,
even if it fails for the other kernels installed on the machine. If
you do not plan to run on the real FPGA board but only use software or
hardware emulation instead, it does not matter if the kernel device
driver is not compiled since it will not be used.

Note that if you want to use a debug version of XRT, use this recipe instead:
```bash
cd Debug
make package
sudo apt install --reinstall ./xrt_201910.2.2.0_19.04-xrt.deb
```

Check that the FPGA board is detected:
```bash
sudo /opt/xilinx/xrt/bin/xbutil flash scan -v
XBFLASH -- Xilinx Card Flash Utility
Card [0]
	Card BDF:		0000:04:00.0
	Card type:		u200
	Flash type:		SPI
	Shell running on FPGA:
		xilinx_u200_GOLDEN_2,[SC=1.8]
	Shell package installed in system:	(None)
	Card name		AU200P64G
	Card S/N: 		2130048BQ00H
	Config mode: 		7
	Fan presence:		P
	Max power level:	75W
	MAC address0:		00:0A:35:05:EF:0E
	MAC address1:		00:0A:35:05:EF:0F
	MAC address2:		FF:FF:FF:FF:FF:FF
	MAC address3:		FF:FF:FF:FF:FF:FF
```


## Install SDAccel

You need the framework to do the hardware synthesis, taking the SPIR
intermediate representation (or HLS C++ or OpenCL or...) and
generating some FPGA configuration bitstream.


Download from
https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/sdaccel-development-environment.html
for example the web installer and use it with
```bash
chmod +x Xilinx_SDAccel_2019.1_0524_1430_Lin64.bin
./Xilinx_SDAccel_2019.1_0524_1430_Lin64.bin
```

Select the `UltraScale+` platform for the U200 and install it in
`/opt/xilinx` (because it is the default location used by XRT too).

Ask for a 30 day trial license for example. It puts some information
into `~/.Xilinx`.

SDAccel comes with some old useless pieces of XRT to
generate/instrospect the bitstream containers and still wants to use
them... So we need to patch SDx to use the right XRT by executing:
```bash
sudo ln -s $XILINX_XRT /opt/xilinx/SDx/2019.1/xrt/xrt-2.1.0-ubuntu19.04
```


## Install the shells for the FPGA board

To execute some kernels on the FPGA board you need a deployment shell
which is an FPGA configuration that pre-defines an architecture on the
FPGA to execute some kernels in some reconfigurable area.

To develop some kernels for your FPGA board and to run some
simulations, you need a development shell that will contain some
internal description specific to the FPGA and its deployment shell so
that the tools can generate the right bitstream for the kernels or the
simulation details.

Pick the latest deployment and development shells from
https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
for your board and for you OS. It might be an older version. For
example it is possible to use a `2018.3` shell with SDAccel `2019.1`,
and a version for Ubuntu `18.04` on a more recent version of Ubuntu or
Debian.

Install the shells:
```bash
dpkg -i xilinx-u200-xdma-201830.1_18.04.deb
dpkg -i xilinx-u200-xdma-201830.1-dev_18.04.deb
```
from where they have been downloaded or adapt the paths.


### Flash the board

If you do not want to use a real board, skip this section.

If you want to use a real board, follow the recipe "Generating the
xbutil flash Command" from
https://www.xilinx.com/html_docs/accelerator_cards/alveo_doc/ftt1547585535561.html
about how to correctly generate the exact flashing command.

Typically you run:
```bash
sudo /opt/xilinx/xrt/bin/xbutil flash scan
XBFLASH -- Xilinx Card Flash Utility
Card [0]
	Card BDF:		0000:04:00.0
	Card type:		u200
	Flash type:		SPI
	Shell running on FPGA:
		xilinx_u200_GOLDEN_2,[SC=1.8]
	Shell package installed in system:	
		xilinx_u200_xdma_201830_1,[TS=0x000000005bece8e1],[SC=3.1]
```
to get the information about the installed shell and you translate this
into a flashing command according to the parameters above:
```bash
sudo /opt/xilinx/xrt/bin/xbutil flash -a xilinx_u200_xdma_201830_1 -t 0x000000005bece8e1 -d 0
XBFLASH -- Xilinx Card Flash Utility
Probing card[0]: Shell on FPGA needs updating
Shell on below card(s) will be updated:
Card [0]
Are you sure you wish to proceed? [y/n]
y
Updating SC firmware on card[0]
INFO: found 5 sections
..............................
INFO: Loading new firmware on SC
.
Updating shell on card[0]
INFO: ***Found 701 ELA Records
Idcode byte[0] ff
Idcode byte[1] 20
Idcode byte[2] bb
Idcode byte[3] 21
Idcode byte[4] 10
Enabled bitstream guard. Bitstream will not be loaded until flashing is finished.
Erasing flash...................................
Programming flash...................................
Cleared bitstream guard. Bitstream now active.
1 Card(s) flashed successfully.
Cold reboot machine to load the new image on FPGA.
```

Unfortunately you need to reboot the machine to have the new shell
loaded inside the FPGA.

Then you can check with a pre-compiled FPGA program provided by the
shell that the board is working correctly with:
```bash
sudo /opt/xilinx/xrt/bin/xbutil validate
INFO: Found 1 cards

INFO: Validating card[0]: xilinx_u200_xdma_201830_1
INFO: Checking PCIE link status: PASSED
INFO: Starting verify kernel test: 
INFO: verify kernel test PASSED
INFO: Starting DMA test
Host -> PCIe -> FPGA write bandwidth = 11127.5 MB/s
Host <- PCIe <- FPGA read bandwidth = 12147.9 MB/s
INFO: DMA test PASSED
INFO: Starting DDR bandwidth test: ..........
Maximum throughput: 47665.777344 MB/s
INFO: DDR bandwidth test PASSED
INFO: Starting P2P test
P2P BAR is not enabled. Skipping validation
INFO: P2P test PASSED
INFO: Starting M2M test
bank0 -> bank1 M2M bandwidth: 12089.7 MB/s	
bank0 -> bank2 M2M bandwidth: 12064.7 MB/s	
bank0 -> bank3 M2M bandwidth: 12084 MB/s	
bank1 -> bank2 M2M bandwidth: 12064.1 MB/s	
bank1 -> bank3 M2M bandwidth: 12066.9 MB/s	
bank2 -> bank3 M2M bandwidth: 12125.8 MB/s	
INFO: M2M test PASSED
INFO: Card[0] validated successfully.

INFO: All cards validated successfully.
```

It looks there is a problem in the Linux `udev` configuration
making the DRM driver inaccessible for some `other` users. A crude
work-around for now is for example to run
```bash
sudo chmod 666 /dev/dri/renderD129
```
if `/dev/dri/renderD129` corresponds to the FPGA board (it might be a
number different from `129`) and check that it is working by running
```bash
clinfo
```
that should expose some OpenCL device parameters for the FPGA board.


## Compile the SYCL compiler

```bash
# Get the source code
git clone --branch sycl/unified/master git@github.com:triSYCL/sycl.git
cd sycl
SYCL_HOME=`pwd`
export XILINX_XRT=/opt/xilinx/xrt
mkdir $SYCL_HOME/build
cd $SYCL_HOME/build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_CXX_STD="c++17" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" \
  -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/sycl \
  -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm-spirv \
  -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" \
  -DLLVM_LINK_LLVM_DYLIB:BOOL=ON \
  $SYCL_HOME/llvm
make -j`nproc` check-all
```

Some checks may fail but that may not be an issue.


## Compiling and running an application

The typical environment is setup with something like
```bash
export XILINX_SDX=/opt/xilinx/SDx/2019.1
export XILINX_VIVADO=/opt/xilinx/Vivado/2019.1
PATH=$PATH:/opt/xilinx/xrt/bin:$XILINX_SDX/bin:$XILINX_VIVADO/bin
# This is the platform of the shell on the FPGA board
export XILINX_PLATFORM=xilinx_u200_xdma_201830_1
export XILINX_XRT=/opt/xilinx/xrt
# Update to the real place the SYCL compiler working tree is:
SYCL_HOME=/var/tmp/rkeryell/SYCL/sycl
SYCL_BIN_DIR=$SYCL_HOME/build/bin
export LD_LIBRARY_PATH=$XILINX_XRT/lib:$SYCL_HOME/build/lib:$LD_LIBRARY_PATH
```

You can compile an application either for real FPGA execution,
software emulation (the SYCL device code is executed by the XRT
runtime on CPU) or hardware emulation (the SYCL device code is
synthesized into RTL Verilog and run by an RTL simulator). Note that
the software and hardware emulation might not work for some system
incompatibility reasons because SDAccel comes with a lot of
system-specific assumptions with a load of old compilers and libraries
instead of just using the ones from the system and the mix-and-match
might be wrong on your current system...

The `XCL_EMULATION_MODE` environment variable selects the compilation &
execution mode and is used by the SYCL compiler and XRT runtime.

So to run an example, for example start with
```bash
cd $SYCL_HOME/sycl/test/xocc_tests/simple_tests
```
- with software emulation:
  ```bash
  export XCL_EMULATION_MODE=sw_emu
  # Configure the simulation environment
  emconfigutil -f $XILINX_PLATFORM --nd 1 --save-temps
  # Compile the SYCL program down to a host fat binary including device code for CPU
  $SYCL_BIN_DIR/clang++ -std=c++2a -fsycl -fsycl-targets=fpga64-xilinx-unknown-sycldevice \
    parallel_for_ND_range.cpp -o parallel_for_ND_range \
    -lOpenCL -I/opt/xilinx/xrt/include
  # Run the software emulation
  ./parallel_for_ND_range
  ```
- with hardware emulation:
  ```bash
  export XCL_EMULATION_MODE=hw_emu
  # Configure the simulation environment
  emconfigutil -f $XILINX_PLATFORM --nd 1 --save-temps
  # Compile the SYCL program down to a host fat binary including the RTL for simulation
  $SYCL_BIN_DIR/clang++ -std=c++2a -fsycl -fsycl-targets=fpga64-xilinx-unknown-sycldevice \
    parallel_for_ND_range.cpp -o parallel_for_ND_range \
    -lOpenCL -I/opt/xilinx/xrt/include
  # Run the hardware emulation
  ./parallel_for_ND_range
  ```
- with real hardware execution on FPGA:
  ```bash
  # Instruct only the compiler about hardware execution (\todo: change this API)
  export XCL_EMULATION_MODE=hw
  # Compile the SYCL program down to a host fat binary including the FPGA bitstream
  $SYCL_BIN_DIR/clang++ -std=c++2a -fsycl -fsycl-targets=fpga64-xilinx-unknown-sycldevice \
    parallel_for_ND_range.cpp -o parallel_for_ND_range \
    -lOpenCL -I/opt/xilinx/xrt/include
  # Unset the variable at execution time to have real execution
  unset XCL_EMULATION_MODE
  # Run on the real FPGA board
  ./parallel_for_ND_range
  ```
Note that the compilation line does not change, just the environment
variable and the use or not of `emconfigutil`.

### Running a bigger example on real FPGA

To run a SYCL translation of
https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/edge_detection
```bash
cd $SYCL_HOME/sycl/test/xocc_tests/sdaccel_ports/vision/edge_detection
export XCL_EMULATION_MODE=hw
$SYCL_BIN_DIR/clang++ -std=c++2a -fsycl \
    -fsycl-targets=fpga64-xilinx-unknown-sycldevice edge_detection.cpp \
    -o edge_detection -lOpenCL `pkg-config --libs opencv` \
    -I/opt/xilinx/xrt/include
unset XCL_EMULATION_MODE
# Execute on one of the images
./edge_detection data/input/eiffel.bmp
```
and then look at the `input.bmp` and `output.bmp` images.

There is another application using a webcam instead, if you have one
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

But if a SYCL program crashes before deallocating the OpenCL buffer
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

Then you need to explicitly deallocate the buffer because the device
driver still has the hope a program wants to use the data of the
allocated buffer in the future...

This can be done by removing the kernel driver and reloading it by
executing:
```bash
sudo rmmod xocl
sudo modprobe xocl
```
