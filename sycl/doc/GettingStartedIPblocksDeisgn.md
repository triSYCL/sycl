Getting started with Vitis ip block design with an AMD/Xilinx FPGA
===================================================================================

Disclaimer: nothing here is supported and this is all about a research
project.

We assume that you are on a recent Linux `x86_64` machine.
We assume that you have installed a recent version for Vitis/Vivado already installed

## Compile the compiler

Building the compiler can be done with Python scripts:

```bash
# Pick some place where The compiler has to be compiled, such as:
git clone --branch VitisIpTarget git@github.com:Ralender/sycl.git
cd sycl
python ./buildbot/configure.py
python ./buildbot/compile.py -t vitis-ip-compiler
```

## Environnement Setup

The only requirement for environnement setup is that there is a vitis_hls in the PATH.
if you already have a setup for this you can use it.

otherwise you need somthing like like
```bash
# Fill the version and path of Vitis/Vivado
export XILINX_VERSION=2021.2
export XILINX_PATH=/path/to/xilinx/root/

export PATH=$PATH:$XILINX_PATH/Vitis_HLS/$XILINX_VERSION/bin
```

### Small examples

create a file named vitis_ip.cpp containing

```cpp
// __VITIS_KERNEL marks the top-level kernel function
__VITIS_KERNEL int test(int a, int b) {
  return a + b;
}
```

then to compile this file use 
```bash
./build/bin/clang++ --target=vitis_ip-xilinx vitis_ip.cpp --vitis-ip-part=xc7vx330t-ffg1157-1 -o a.zip
```

 ``--target=vitis_ip-xilinx`` specifies that we are targeting vitis ip blocks
 ``--vitis-ip-part=xc7vx330t-ffg1157-1`` specifies which Xilinx device we are targeting ``xc7vx330t-ffg1157-1`` needs to be replace by the part id you are working with

 The output is a zip file that can be loaded into Vivado when making a block design

### Further Reading

see documentation about the SYCL flow [here](GettingStartedXilinxFPGA.md)
