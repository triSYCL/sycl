#!/bin/bash
# A small convenience script for compiling and running all the tests for FPGA
# sw_emu and hw_emu. As it gets a little tiring and time consuming manually
# doing it. This script is likely to take a very long time to finish if you run
# everything. Upwards of a few hours.
#
# The most important tests to run and make sure still function when merging
# upstream are:
#   id_mangle
#   math_mangle
#   parallel_for
#   reqd_work_group_size
#   single_task_vector_add
#   edge_detection
#   internal_defines
#
# These test are the "main" functionality that could break on upstream merge,
# most of the rest is icing on the cake tests that are related to testing SYCL
# functionality. They're also important, but larger breaking issues are more
# likely to show in one of the above tests.
#
# Currently most of the tests will assert when they fail which will stop this
# test shell script running as it sends a SIGABRT. These tests could perhaps be
# changed to something like a SYCL runtime throw that's caught and prints an
# error message. This would perhaps be more user friendly and make this script
# more useful long term.
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
# Note: You will of course have to have your environment setup to compile for
# Xilinx FPGA, e.g. XILINX_XRT set and a useable version of SDAccel with all the
# environment it requires to compile one of our tests.
#
# TODO: Convert this test script to something more robust like the LLVM test
# suite. If it's the LLVM test suite that we use, we should make it
# a non-default option to compile and run the FPGA tests. As it's probably not a
# great idea for us to force users to compile several hours of tests any time
# they wish to test regular check-all or even base host or Intel SYCL
# functionality.

export SYCL_PI_TRACE=127
export XPTI_TRACE_ENABLE=1

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
# this is just to make sure we didn't regress our own tests, you should also
# run make check-all or make check-sycl from the build directory:
# check-all - if there was any compiler modifications required
# check-sycl - if it was just some SYCL runtime modifications
DEFAULT_INTEL_ARGS=(-std=c++2a -fsycl -lOpenCL)

# $1 Test File Name
# $2 The tests directory
# $3 XCL_EMULATION_MODE
# $4 Extra Args for Compilation, supplied as a ref to array
# $5 Extra Args for Execution, supplied as a ref to array
run_test () {
  export XCL_EMULATION_MODE=$3

  USED_DEFAULT_ARGS=("${DEFAULT_XFPGA_ARGS[@]}")
  if [ $3 = "intel" ]; then
    USED_DEFAULT_ARGS=("${DEFAULT_INTEL_ARGS[@]}")
  fi

  # get reference to an array, even if it's 1 argument, it should be passed in
  # as an array to make life simpler and avoid a lot of irritating logic
  if [ ! -z "$4" ]; then
    local -n COMPILATION_ARG_ARR_REF=$4
  else
    local COMPILATION_ARG_ARR_REF=""
  fi

  if [ ! -z "$5" ]; then
    local -n RUNTIME_ARG_ARR_REF=$5
  else
    local RUNTIME_ARG_ARR_REF=""
  fi

  echo "" >>  $TEST_OUTPUT_FILE
  echo "" >>  $TEST_OUTPUT_FILE
  echo "Compiling $2/$1.cpp" >>  $TEST_OUTPUT_FILE
  $CLANG_BIN/clang++ -g3 "${USED_DEFAULT_ARGS[@]}" "${COMPILATION_ARG_ARR_REF[@]}" \
    "$2/$1.cpp" -o "$2/$1.$XCL_EMULATION_MODE" >> $TEST_OUTPUT_FILE 2>&1

  # The default is not to run for hardware (hw), to run tests for hardware
  # remove the if block, unset XCL_EMULATION_MODE and it should make an attempt
  # to execute the test although you will of course require the correct piece of
  # hardware
  if [[ $3 != "hw" ]]; then
    echo "" >>  $TEST_OUTPUT_FILE
    echo "" >>  $TEST_OUTPUT_FILE
    echo "Executing $2/$1.cpp" >> $TEST_OUTPUT_FILE
    ./"$2/$1.$XCL_EMULATION_MODE" "${RUNTIME_ARG_ARR_REF[@]}" \
      >> $TEST_OUTPUT_FILE 2>&1 && echo "passed" || echo "failed: $?" >>  $TEST_OUTPUT_FILE
  fi

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
 run_test "math_mangle" "simple_tests" "$1" # failed infinitloop
 run_test "multi_parallel_for" "simple_tests" "$1"
 run_test "parallel_for" "simple_tests" "$1"
 run_test "reqd_work_group_size" "simple_tests" "$1" # failed infinitloop
#  Note: There appears to be a race condition in hw_emu for
#  single_task_vector_add, sometimes passes sometimes fails.
 run_test "single_task_vector_add" "simple_tests" "$1"
 run_test "vector_math" "simple_tests" "$1" # failed infinitloop
 run_test "simple_struct" "simple_tests" "$1"
 run_test "ternary_compare" "simple_tests" "$1"
 run_test "kernel_uint_name" "simple_tests" "$1"

  emconfigutil -f $XILINX_PLATFORM --od sdaccel_ports/vision/edge_detection

# This test is a bit of a monster for time consumption when run in hw_emu,
# if it compiles and runs the first few iterations...it's a success
  COMPILER_ARG_ARR=(`pkg-config --libs --cflags opencv4`)
  RUNTIME_ARG_ARR=(sdaccel_ports/vision/edge_detection/data/input/eiffel.bmp)
#  run_test "edge_detection" "sdaccel_ports/vision/edge_detection" "$1" \
#    COMPILER_ARG_ARR RUNTIME_ARG_ARR
}

# compile and test for intel, don't want to break existing functionality.
# test_list "intel"

# Compile and Run Tests for Software and Hardware Emulation
# test_list "sw_emu"
test_list "hw_emu"
# I would advise only running this on a subset of the tests or if you have a
# weekend to run the tests.
# The sycl-xocc script doesn't play well with multiple parallel invocations of
# the compiler due to the way it treats temporary files right now. At the very
# least it's not been rigorously tested. Currently the script is improved via
# lazy evolution, if its required it's added on..  This script also does not
# execute the tests, it will just compile them.
# test_list "hw"

