#ifndef SYCL_XILINX_KERNEL_PROPERTIES_HPP
#define SYCL_XILINX_KERNEL_PROPERTIES_HPP

namespace cl::sycl::xilinx {
  template <int DimX, int DimY, int DimZ>
  struct reqd_work_group_size {
    static constexpr int x = DimX;
    static constexpr int y = DimY;
    static constexpr int z = DimZ;
  };
}

#endif // SYCL_XILINX_KERNEL_PROPERTIES_HPP
