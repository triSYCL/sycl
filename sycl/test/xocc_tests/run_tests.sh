#!/bin/bash
# A small convenience script for compiling and running all the tests for FPGA
# sw_emu and hw_emu. As it gets a little tiring and time consuming manually
# doing it. This script is likely to take a very longtime to finish.
#
# This script won't compare the results or give you feedback on the end result
# of the tests. It will spit the output into a file which you'll unfortunately
# have to visually parse for now. It's not intended to be a replacement for
# the more complex and useful LLVM tests. It's a stopgap script till time can be
# put into integrating these tests into the suite and adding the option of
# running all the other LLVM SYCL tests for FPGA (i.e the ones that Intel and
# other contributors have made).
#
# Example invocation:
# ./run_tests.sh /my/clang/build/bin optional_output_file.txt
#
# Note: You will of course have to have your enviornment setup to compile for
# Xilinx FPGA, e.g. XILINX_XRT set and a useable version of SDAccel with all the
# environment it requires to compile one of our tests.
#
# TODO: Convert this test script to something more robust like the LLVM test
# suite. If it's the LLVM test suite that we use, we should make it
# a non-default option to compile and run the FPGA tests. As it's probably not a
# great idea for us to force users to compile several hours of tests any time
# they wish to test regular check-all or even base host or Intel SYCL
# functionality.

usage() { echo run_tests: error: $2 >&2; exit $1; }

# The directory that SYCL Clang resides in
if [[ -z "$1" ]]; then
  usage 1 "no directory containing Clang specified"
fi

# Either create a new file named test_dump.txt or create one with a name given
# to the script
TEST_OUTPUT_FILE=test_dump.txt
if [[ -z "$2" ]]; then
  cat /dev/null > test_dump.txt
else
  cat /dev/null > $2
  TEST_OUTPUT_FILE=$2
fi

if [[ -z "$XILINX_XRT" ]]; then
  usage 2 "no XILINX_XRT environment variable set, please specify XRT directory"
fi

CLANG_BIN=$1
DEFAULT_XFPGA_ARGS=(-std=c++2a -fsycl -fsycl-targets=fpga64-xilinx-unknown-sycldevice -I$XILINX_XRT/include/ -lOpenCL)

# $1 Test File Name
# $2 The tests directory
# $3 XCL_EMULATION_MODE
# $4 Extra Args for Compilation
# $5 Extra Args for Execution
run_test () {
  export XCL_EMULATION_MODE=$3

  echo "" >>  $TEST_OUTPUT_FILE
  echo "" >>  $TEST_OUTPUT_FILE
  echo "Compiling $2/$1.cpp" >>  $TEST_OUTPUT_FILE
  $CLANG_BIN/clang++ "${DEFAULT_XFPGA_ARGS[@]}" `pkg-config --libs opencv` "$4" "$2/$1.cpp" \
    -o "$2/$1.$XCL_EMULATION_MODE" >> $TEST_OUTPUT_FILE 2>&1

  echo "" >>  $TEST_OUTPUT_FILE
  echo "" >>  $TEST_OUTPUT_FILE
  echo "Executing $2/$1.cpp" >> $TEST_OUTPUT_FILE
  ./"$2/$1.$XCL_EMULATION_MODE" "$5" >> $TEST_OUTPUT_FILE 2>&1

  echo "" >>  $TEST_OUTPUT_FILE
  echo "" >>  $TEST_OUTPUT_FILE
}

# $1 The emulation mode you wish to compile and execute the list of tests with
test_list () {
  emconfigutil -f $XILINX_PLATFORM --od simple_tests

  run_test "accessor_copy" "simple_tests" "$1"
  run_test "explicit_copy" "simple_tests" "$1"
  run_test "constexpr_correct" "simple_tests" "$1"
  run_test "id_mangle" "simple_tests" "$1"
  run_test "integration_header_check" "simple_tests" "$1"
  run_test "internal_defines" "simple_tests" "$1"
  run_test "math_mangle" "simple_tests" "$1"
  run_test "multi_parallel_for_ND_range" "simple_tests" "$1"
  run_test "parallel_for_ND_range" "simple_tests" "$1"
  run_test "reqd_work_group_size" "simple_tests" "$1"
  run_test "single_task_vector_add" "simple_tests" "$1"
  run_test "vector_math" "simple_tests" "$1"

  emconfigutil -f $XILINX_PLATFORM --od sdaccel_ports/vision/edge_detection

  run_test "edge_detection" "sdaccel_ports/vision/edge_detection" "$1" \
    "" "sdaccel_ports/vision/edge_detection/data/input/eiffel.bmp"
}

# Compile and Run Tests for Software and Hardware Emulation
test_list "sw_emu"
test_list "hw_emu"

