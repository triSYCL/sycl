// This is auto-generated SYCL integration header.

#include <CL/sycl/detail/kernel_desc.hpp>

// Forward declarations of templated kernel function types:
class krnl_sobel;

namespace cl {
namespace sycl {
namespace detail {

// names of all kernels defined in the corresponding source
static constexpr
const char* const kernel_names[] = {
  "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10krnl_sobel"
};

// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10krnl_sobel
  { kernel_param_kind_t::kind_std_layout, 4, 0 },

};

// indices into the kernel_signatures array, each representing a start of
// kernel signature descriptor subarray of the kernel_signatures array;
// the index order in this array corresponds to the kernel name order in the
// kernel_names array
static constexpr
const unsigned kernel_signature_start[] = {
  0 // _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10krnl_sobel
};

// Specializations of this template class encompasses information
// about a kernel. The kernel is identified by the template
// parameter type.
template <class KernelNameType> struct KernelInfo;

// Specializations of KernelInfo for kernel function types:
template <> struct KernelInfo<class krnl_sobel> {
  static constexpr const char* getName() { return "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE10krnl_sobel"; }
  static constexpr unsigned getNumParams() { return 1; }
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+0];
  }
};

} // namespace detail
} // namespace sycl
} // namespace cl

