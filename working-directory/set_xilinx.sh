#!/bin/bash
# Should be invoked using source (or .) e.g. source set_xilinx.sh

# self compiled XRT version
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
export PATH=$XILINX_XRT/bin:$PATH

# path to platforms
export PLATFORM_REPO_PATHS=/proj/xbuilds/2018.3_qualified_latest/xbb/dsadev/opt/xilinx/platforms

# Set emulation mode to sw_emu (can be empty for no emu or have hw_emu instead)
export XCL_EMULATION_MODE=sw_emu

# Use Alexandre's settings64.sh to setup enviornment variables for the debug version
# of HLS/VIVADO (if this doesn't exist you can use the settings64 from the daily releases
# they are however not as good and comprehensive)
SAVE_DIR=$(pwd)
cd /net/xsjswsvm1-lif1/hdstaff1/isoard/Perforce/Rodin/REL/2018.3/
source settings64.sh


# Generate files and platform for a mock device (the json config file generated from this should be accessible for programs you run: e.g. same directory, otherwise it will find
# the default device), this must be executed in a directory you have read and write access to. The --nd 1 is not required, it simply states the number of devices.
cd $SAVE_DIR
emconfigutil -f xilinx_vcu1525_dynamic_5_1 --nd 1
