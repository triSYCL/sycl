# Test Cases

This repository contains the same tests as the Intel SYCL repository, however
the majority of these tests have yet to be tested when compiling for Xilinx
FPGA's. So don't be too surprised if a test doesn't work quite as anticipated,
but please do open an issue on it.

Compiling and executing the test suite using Xilinx FPGA's and the Xilinx
compilation flow is also yet to be integrated, so when you're running
check-all your testing SYCL with another target like an Intel CPU usually!
Integration with the full test suite is something we aim for eventually however.

For now we have a test directory called xocc_tests, which contains tests that
have been tested and shown to either run or not run. Test's that aren't quite
working yet are in the issue_related directory.

## Test Directories of xocc_tests

All the tests in the simple_tests directory are small unit tests with the
intention of testing that certain functionality works for the Xilinx tool chain.

The tests in sdaccel_ports are rough implementations of tests in the
[SDAccel Examples repository](https://github.com/Xilinx/SDAccel_Examples)
repository.

The issue_related directory is mostly for development purposes to hold test
cases that are known to have problems, either because of unclear specification
behavior, compiler or runtime bugs. The bugs can be specific to this project
or the SYCL implementation as a whole. Once the issues are fixed the test cases
should be moved into simple_tests where possible.

The utilities directory is mostly for abstractions or functionality that's
used inside the test cases but has no place in the SYCL runtime.

Feel free to contribute other tests that you find interesting or that come
alongside issues you have encountered and/or fixed. We're also happy to take
in more complex examples!

## Commands

These tests should all be executable in conjunction with the Xilinx SDAccel and
Vivado tool chains alongside an installation of the XRT runtime. A more
comprehensive setup explanation can be found in
[XilinxFPGACompilation.md](XilinxFPGACompilation.md)

Most of the tests in this folder are executable using either Xilinx or Intel
OpenCL supporting hardware based on the following two compiler commands (just
be-aware that some tests still use hard-coded selectors for now):

### Xilinx Compile Command:

```bash
$SYCL_BIN_DIR/clang++ -std=c++2a -fsycl \ 
  -fsycl-targets=fpga64-xilinx-unknown-sycldevice test-case.cpp \ 
  -o test-case -lOpenCL
```

### Regular Compile Command:

```bash
$SYCL_BIN_DIR/clang++ -std=c++2a -fsycl test-case.cpp -o test-case \
  -lOpenCL
```

If a test requires a more complex incantation than the above then it should be
stated at the top of the file in the comment.

## Notes

These tests have only been run on hardware emulation and software emulation so
far and not on actual hardware.
