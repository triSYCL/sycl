Getting started with Vitis IP block design with an AMD/Xilinx FPGA
===================================================================================

Disclaimer: nothing here is supported and this is all about a research
project.

We assume that you are on a recent Linux `x86_64` machine. with a recent C++ compiler, python3 and Vitis/Vivado already installed.

## Compile the compiler

Building the compiler can be done with Python scripts:

```bash
# Pick some place where the compiler has to be compiled, such as:
git clone --branch VitisIpTarget https://github.com/Ralender/sycl.git
cd sycl
python ./buildbot/configure.py
python ./buildbot/compile.py -t vitis-ip-compiler
```

## Environnement Setup

The requirements for environment setup is that there is a vitis_hls in the PATH and the libraries shipped with Vitis are in the LD_LIBRARY_PATH.
if you already have a setup for this you can use it.

Otherwise you will need to make a setup.sh script like this:
```bash
# Fill the version
XILINX_VERSION=2021.2
# Fill the path to the root of the installation, the directory containing Vitis, Vivado and Vitis_HLS
XILINX_PATH=/path/to/vitis/root

# This it to have vitis_hls in the PATH
export PATH=$PATH:$XILINX_PATH/Vitis_HLS/$XILINX_VERSION/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XILINX_PATH/Vitis_HLS/$XILINX_VERSION/lib/lnx64.o
```

### Small examples

create a file named `vitis_ip.cpp` containing

```cpp
// __VITIS_KERNEL marks the top-level kernel function
__VITIS_KERNEL int test(int a, int b) {
  return a + b;
}
```

then, to compile this file, use 
```bash
./build/bin/clang++ --target=vitis_ip-xilinx vitis_ip.cpp --vitis-ip-part=xc7vx330t-ffg1157-1 -o adder.zip
```

 `--target=vitis_ip-xilinx` specifies that we are targeting vitis ip blocks
 `--vitis-ip-part=xc7vx330t-ffg1157-1` specifies which Xilinx device we are targeting `xc7vx330t-ffg1157-1` needs to be replaced by the part id you are working with.

 The output is a Vivado IP zip archive that can be loaded into Vivado when making a block design

### Further Reading

see documentation about the SYCL flow [here](GettingStartedXilinxFPGA.md)
