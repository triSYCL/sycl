/*
 No device compiler command, just cpu/host x86, no cross compilation:
  clang++ -std=c++2a hello_world.cpp -I/ACAP++/acappp/include \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs`

  Cross compilation for ARM host CPU/PS, no device compilation:
    So specify target to be aarch64, specify that the CPU type is a72 and that
    we want clang to consider our arm images root as it's root file system.
    This makes life much simpler than it could be.. the rest is normal ACAP++
    includes
  $ISYCL_BIN_DIR/clang++ -std=c++2a -target aarch64-linux-gnu -mcpu=cortex-a72 \
    --sysroot /net/xsjsycl41/srv/Ubuntu-19.04/arm64-root-server-rw-tmp/ \
    `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs` \
    -I/ACAP++/acappp/include  hello_world.cpp

  Cross compilation for ARM host CPU/PS with AIEngine device compilation:
     -fforce-enable-int128 is a work around to avoid some compilation issues
     that occur in system headers and boost when trying to cross compile. This
     is quite probably a misconfiguration of the TargetInfo/TargetOptions or
     a side affect of using a 64 bit arm root system to offload to a 32 bit
     system. This should be looked into in a little more detail in the future,
     but in reality the TargetInfo shouldn't really default to SPIR32/64 it
     should probably be a little tailored to FPGA or the AIE to make life a
     little easier where possible.
     -aux-triple should be equal to the triple as without it their is a disconect
     in the CompilerInstance/CompilerInvocation where it will use the triple of
     the system that's compiling the binary. For example if you invoke this
     without the aux-triple on an x86 you'll end up with some x86 compiler
     definitions and builtins as the InitPreprocessor will ignore the -target
     and prioritize the aux-triple, which when no defined defaults to the
     compiling system
  $ISYCL_BIN_DIR/clang++ -std=c++2a -Xclang -fforce-enable-int128 \
  -Xclang -aux-triple -Xclang aarch64-linux-gnu \
  -target aarch64-linux-gnu -mcpu=cortex-a72 -fsycl \
  -fsycl-targets=aie32-xilinx-unknown-sycldevice \
  -fsycl-header-only-library \
  --sysroot /net/xsjsycl41/srv/Ubuntu-19.04/arm64-root-server-rw-tmp/ \
  `pkg-config gtkmm-3.0 --cflags` `pkg-config gtkmm-3.0 --libs` \
  -I/ACAP++/acappp/include  hello_world.cpp

*/

#include <iostream>
// This is ACAP SYCL, not the Intel-triSYCL SYCL implementation full merging of
// the implementations is a little too complex for the timescale we have
#include <sycl.hpp>

using namespace sycl::vendor::xilinx;

template <typename AIE, int X, int Y>
struct prog : acap::aie::tile<AIE, X, Y> {
  /// The run member function is defined as the tile program
  void run() { // maybe try printf see if optimized out
    std::cout << "Hello one processing element \n";
//    printf("Hello one processing element \n");
  }
};

int main() {
  acap::aie::array<acap::aie::layout::one_pe, prog>{}.run();

  return 0;
}
