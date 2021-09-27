// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: %ACC_RUN_PLACEHOLDER %t.out

//
// Regression test, based on https://github.com/triSYCL/sycl/issues/64
// submitted by j-stephan.
//
// This was an existing issue where integration header appeared to lose some
// type information for std types like uint32_t. It was a problem shared across
// Intel/Xilinx Implementations and is now fixed in at least this particular
// instance.

#include <cstdint>
#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"


template <std::uint32_t Var>
struct foo
{
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc;

    auto operator()() const
    {
        for(auto i = 0; i < 1024; ++i)
            acc[i] = Var;
    }
};

auto main() -> int
{
    selector_defines::CompiledForDeviceSelector selector;

    auto queue = cl::sycl::queue{selector};

    auto buf = cl::sycl::buffer<int>{cl::sycl::range<1>{1024}};

    queue.submit([&](cl::sycl::handler& cgh)
    {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

        auto kernel = foo<42>{acc};
        cgh.single_task<foo<42>>(kernel);
    });

    auto rb = buf.get_access<cl::sycl::access::mode::read>();

    for (int i = 0; i < buf.size(); ++i) {
      assert(rb[i] == 42 && " execution of kernel is invalid");
    }
}
