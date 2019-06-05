// RUN: true

/*
  An example based on the SYCL Specifications explicitcopy.cpp example, tests
  that the Handlers copy method is working as intended.
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
