// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

//
// Regression test, based on https://github.com/triSYCL/sycl/issues/64
// submitted by j-stephan.
//
// This was an existing issue where integration header appeared to lose some
// type information for std types like uint32_t. It was a problem shared across
// Intel/Xilinx Implementations and is now fixed in at least this particular
// instance.

#include <cstdint>
#include <sycl/sycl.hpp>



template <std::uint32_t Var>
struct foo
{
    sycl::accessor<int, 1, sycl::access::mode::read_write> acc;

    auto operator()() const
    {
        for(auto i = 0; i < 1024; ++i)
            acc[i] = Var;
    }
};

auto main() -> int
{
    auto queue = sycl::queue{};

    auto buf = sycl::buffer<int>{sycl::range<1>{1024}};

    queue.submit([&](sycl::handler& cgh)
    {
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);

        auto kernel = foo<42>{acc};
        cgh.single_task<foo<42>>(kernel);
    });

    auto rb = buf.get_access<sycl::access::mode::read>();

    for (int i = 0; i < buf.size(); ++i) {
      assert(rb[i] == 42 && " execution of kernel is invalid");
    }
}
