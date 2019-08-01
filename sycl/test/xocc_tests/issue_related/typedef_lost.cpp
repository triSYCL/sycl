// RUN: true
//
// Existing issue where integration header appears to lose some type information
// or more likely it just is never defined for host execution.
// It doesn't appear to be a problem specific to the Xilinx implementation,
// it seems like a shared problem.
#include <cstdint>
#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"


template <std::uint32_t Var>
struct foo
{
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc;

    auto operator()()
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

    queue.wait();
}
