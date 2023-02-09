#include <iostream>
#include <sycl/sycl.hpp>
int main() {
  sycl::buffer<int> v { 10 };
  auto run = [&] (auto device_name, auto work) {
    sycl::queue { [&](sycl::device dev) {
      return (device_name == dev.template get_info<sycl::info::device::name>()) - 1;
    } }.submit([&](auto& h) {
      auto a = sycl::accessor { v, h };
      h.parallel_for(a.size(), [=](int i) { work(i, a); });
    });
  };
  run("Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz", [](auto i, auto a) { a[i] = i; });
  run("Quadro P400", [](auto i, auto a) { a[i] = 2 * a[i]; });
  run("Intel(R) FPGA Emulation Device", [](auto i, auto a) { --a[i]; });
  run("AMD Radeon VII", [](auto i, auto a) { a[i] = a[i] * a[i]; });
  run("xilinx_u200_gen3x16_xdma_base_1", [](auto i, auto a) { a[i] += + 3; });
  for (auto e : sycl::host_accessor { v })
    std::cout << e << ", ";
  std::cout << std::endl;
}
/*
  To compile, according to the available devices, for example with
  $DPCPP_HOME/llvm/build/bin/clang++ -std=c++2b -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda,amdgcn-amd-amdhsa,fpga64_hls_hw,spir64_fpga -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx906 -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_61 inclusive_devices.cpp -o inclusive_devices
*/
