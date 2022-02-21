// REQUIRES: xocc, spir
// XFAIL: hw

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*
  An example based on the SYCL Specifications explicitcopy.cpp example, tests
  that the Handlers copy method is working as intended.

  There is a much more complex example testing all the variations of copy and
  fill named: handler_mem_op.cpp, most of the use cases in the test work except
  the 2D/3D Accessor copy tests, which XRT throws the following errors for:

  [XRT] ERROR: src_origin,region,src_row_pitch,src_slice_pitch out of range
  [XRT] ERROR: src_origin,region,src_row_pitch,src_slice_pitch out of range
  [XRT] ERROR: src_origin,region,src_row_pitch,src_slice_pitch out of range
  OpenCL API failed. /storage/ogozillo/intel-sycl/sycl/sycl/source/detail/
    memory_manager.cpp:282: OpenCL API returns: -30 (CL_INVALID_VALUE)
  [XRT] ERROR: src_origin,region,src_row_pitch,src_slice_pitch out of range
  [XRT] ERROR: Internal error. cl_mem doesn't map to buffer object
  [XRT] ERROR: Internal error. cl_mem doesn't map to buffer object
  [XRT] ERROR: Internal error. cl_mem doesn't map to buffer object
  OpenCL API failed. /storage/ogozillo/intel-sycl/sycl/sycl/source/detail/
    memory_manager.cpp:243: OpenCL API returns: -6 (CL_OUT_OF_HOST_MEMORY)
  [XRT] ERROR: Internal error. cl_mem doesn't map to buffer object
  terminate called after throwing an instance of 'cl::sycl::runtime_error'
  free(): corrupted unsorted chunks

  \TODO fix the 2D/3D Copy for XRT
*/

#include <numeric>
#include <vector>

#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

int main()
{
  selector_defines::CompiledForDeviceSelector selector;
  auto q = queue{selector};

  const size_t nElems = 10u;

  std::vector<int> v(nElems);
  std::iota(std::begin(v), std::end(v), 0);

  buffer<int, 1> b{cl::sycl::range<1>(nElems)};

  q.submit([&](handler& cgh) {
    accessor<int, 1, access::mode::write, access::target::global_buffer>
      acc(b, cgh, range<1>(nElems / 2), id<1>(0));

    cgh.copy(v.data(), acc);
  });

  auto acc_r = b.get_access<access::mode::read>();

  for (int i = 0; i < nElems / 2; ++i) {
    assert(acc_r[i] == i);
  }

  return 0;
}
