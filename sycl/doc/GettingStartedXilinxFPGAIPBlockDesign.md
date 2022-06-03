Getting started with Vitis IP block design with an AMD/Xilinx FPGA
===================================================================================

Disclaimer: nothing here is supported and this is all about a research
project.

This describes the C++20 non-single source Vitis IP mode compilation
flow used to generate Vitis IP which can be used later into Vivado for
further integration in a project.

This uses some features developed for the SYCL compiler to provide
something similar to HLS C++ but with more modern C++20.

We assume that you are on a recent Linux `x86_64` machine. with a
recent C++ compiler, `python3` and Vitis/Vivado already installed.


## Compile the compiler

Building the compiler can be done with Python scripts:

```bash
# Pick some place where the compiler has to be compiled, such as:
SYCL_HOME=~/sycl_workspace
mkdir $SYCL_HOME
cd $SYCL_HOME
# You can also try --branch sycl/unified/next for a bleeding edge experience
git clone --branch sycl/unified/master https://github.com/triSYCL/sycl llvm
# Minimum configuration, but you can add more
python $SYCL_HOME/llvm/buildbot/configure.py
# Compile only the minimum for Vitis IP support. If you compile for
# the full SYCL support, it will work too but the compilation of the
# tool is slower.
python $SYCL_HOME/llvm/buildbot/compile.py --build-target vitis-ip-compiler
```


## Environnement Setup

The requirements for environment setup is that there is a `vitis_hls` in
the PATH and the libraries shipped with Vitis are in the
LD_LIBRARY_PATH.

if you already have a setup for this you can use it.

Otherwise you will need to make a setup.sh script like this:
```bash
# The place where SYCL has been compiled:
SYCL_HOME=~/sycl_workspace
# Fill the version
XILINX_VERSION=2021.2
# Where all the AMD/Xilinx tools are
XILINX_ROOT=/opt/xilinx
# Where the SYCL compiler binaries are:
SYCL_BIN_DIR=$SYCL_HOME/llvm/build/bin
XILINX_VITIS_HLS=$XILINX_ROOT/Vitis_HLS/$XILINX_VERSION
XILINX_VIVADO=$XILINX_ROOT/Vivado/$XILINX_VERSION
# Add the various tools in the PATH
PATH=$PATH:$SYCL_BIN_DIR:$XILINX_VITIS_HLS/bin:$XILINX_VIVADO/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XILINX_VITIS_HLS/lib/lnx64.o:$SYCL_HOME/llvm/build/lib
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
# Be sure to use the right clang++
$SYCL_BIN_DIR/clang++ --target=vitis_ip-xilinx vitis_ip.cpp --vitis-ip-part=xc7vx330t-ffg1157-1 -o adder.zip
```

 `--target=vitis_ip-xilinx` specifies that we are targeting vitis ip blocks
 `--vitis-ip-part=xc7vx330t-ffg1157-1` specifies which Xilinx device we are targeting `xc7vx330t-ffg1157-1` needs to be replaced by the part id you are working with.

 The output is a Vivado IP zip archive that can be loaded into Vivado when making a block design


### Further Reading

see documentation about the SYCL flow [here](GettingStartedXilinxFPGA.md)
