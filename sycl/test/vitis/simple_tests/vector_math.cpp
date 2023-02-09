// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

/*
   Modified version of sycl/test/built-ins/vector_math.cpp
   
   Used to run a few small vector tests that will hopefully be expanded on in the future.
   
   The vector fmin overload currently seems to be one of the problem builtins that don't correspond to the exact mangling. 
*/

#include <sycl/sycl.hpp>

#include <array>
#include <cassert>


using namespace sycl;

int main() {
  queue q;

  // fmin, missng...? _Z4fminDv2_fDv2_f
/*
  {
    sycl::cl_float2 r{0};
    {
      buffer<sycl::cl_float2, 1> BufR(&r, range<1>(1));

      q.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fminF2F2>([=] {
          AccR[0] = sycl::fmin(sycl::cl_float2{0.5f, 3.4f},
                                   sycl::cl_float2{2.3f, 0.4f});
        });
      });
    }
    sycl::cl_float r1 = r.x();
    sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 0.5f);
    assert(r2 == 0.4f);
  }
*/
  // native::exp
  {
    sycl::cl_float2 r{0};
    {
      buffer<sycl::cl_float2, 1> BufR(&r, range<1>(1));
      q.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class nexpF2>([=] {
          AccR[0] = sycl::native::exp(sycl::cl_float2{1.0f, 2.0f});
        });
      });
    }
    sycl::cl_float r1 = r.x();
    sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 > 2.718 && r1 < 2.719); // ~2.718281828459045
    assert(r2 > 7.389 && r2 < 7.390); // ~7.38905609893065
  }

  // fabs
  {
    sycl::cl_float2 r{0};
    {
      buffer<sycl::cl_float2, 1> BufR(&r, range<1>(1));
      q.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class fabsF2>([=] {
          AccR[0] = sycl::fabs(sycl::cl_float2{-1.0f, 2.0f});
        });
      });
    }
    sycl::cl_float r1 = r.x();
    sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // floor
  {
    sycl::cl_float2 r{0};
    {
      buffer<sycl::cl_float2, 1> BufR(&r, range<1>(1));
      q.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class floorF2>([=] {
          AccR[0] = sycl::floor(sycl::cl_float2{1.4f, 2.8f});
        });
      });
    }
    sycl::cl_float r1 = r.x();
    sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // ceil
  {
    sycl::cl_float2 r{0};
    {
      buffer<sycl::cl_float2, 1> BufR(&r, range<1>(1));
      q.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class ceilF2>([=] {
          AccR[0] = sycl::ceil(sycl::cl_float2{1.4f, 2.8f});
        });
      });
    }
    sycl::cl_float r1 = r.x();
    sycl::cl_float r2 = r.y();
    std::cout << "r1 " << r1 << " r2 " << r2 << std::endl;
    assert(r1 == 2);
    assert(r2 == 3);
  }

  return 0;
}
