// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// XFAIL
// Disable because it contains a deadlock

/*
  This test is intended to check the LLVM IR output for correctness
  (reqd_work_group_size attribute is applied to the kernel via meta-data) and to
  see that the reqd_work_group_size property is appropriately picked up and
  applied in a number of different use cases. It is of note that only the first
  application of it on a kernel_name is chosen rather than any subsequent
  application. A future direction would possibly be to emit a compile time
  diagnostic in these cases however.

  As well as to test that when executed for a Xilinx device it will break
  the run-time appropriately (which indicates that the compiler attribute is
  being correctly applied even if the user is misusing it and breaking the
  run-time) and executes appropriately for Xilinx devices in the cases where it
  is appropriately applied (in this case the test is just that the correct
  kernels execute).

  The kernels that should fail on execution for Xilinx devices are numbers 6,7,8
  and 9. Compiling without XOCC/Xilinx related flags results in the property
  being ignored for now.

  You can also run this with Intel devices (provided std C++17), but the test is
  meaningless apart from testing that the vendor extension Xilinx
  reqd_work_group_size doesn't break the Intel compilation/run-time.

  The underlying OpenCL enqueue call is clEnqueueNDRangeKernel for both
  single_task and parallel_for. So the xilinx::reqd_work_group_size property
  needs to meet the following rules as it's essentially the reqd_work_group_size
  attribute:

  1) CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and number of
     work-items specified by global_work_size is not evenly divisible by size of
     work-group given by local_work_size or does not match the work-group size
    specified for kernel using the __attribute__ ((reqd_work_group_size(X, Y,
    Z))) qualifier in program source.

  2) CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the total
     number of work-items in the work-group computed as local_work_size[0] *...
     local_work_size[work_dim - 1] is greater than the value specified by
     CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL Device Queries for
     clGetDeviceInfo.

  3) CL_INVALID_WORK_GROUP_SIZE if local_work_size is NULL and the
    __attribute__((reqd_work_group_size(X, Y, Z))) qualifier is used to declare
    the work-group size for kernel in the program source.

  If these rules are not met the run-time should appropriately throw the OpenCL
  CL_INVALID_WORK_GROUP_SIZE error.
*/

#include <CL/sycl.hpp>
#include <iostream>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

// Forward declaring some class names that are similar to the property name to
// make sure that it's not inaccurately chosen
class reqd_work_group_size_test;
class reqd_work_group_size_test2;
class reqd_work_group_size_test3;
class reqd_work_group_size_test4;
class reqd_work_group_size_test5;
class reqd_work_group_size_test6;
class reqd_work_group_size_test7;
class reqd_work_group_size_test8;
class reqd_work_group_size_test9;
class reqd_work_group_size_test10;

// Just using this to make sure the llvm pass doesn't accidentally pick up the
// incorrect thing
__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl::xilinx {
  template <int DimX, typename T>
  struct conflict_test {};
}

}

int main() {
  selector_defines::CompiledForDeviceSelector selector;
  queue q {selector};

  range<3> k_range {8,8,8};
  buffer<unsigned int> ob(range<1>{1});

  auto wb = ob.get_access<access::mode::write>();
  wb[0] = 0;

  std::cout << "reqd_work_group_size_test begin (xilinx fail, intel pass) \n";

  {
    // This is an illegal use-case of reqd_work_group_size, there is no local
    // work group size specified to this form of parallelism construct and we
    // do not wish to enforce reqd_work_group_size in the runtime instead. As
    // without access to an nd_item the user cannot access the local space.
    // There is no SYCL rules for this, however in the context of OpenCL no
    // defined local work group size is illegal when using the
    // reqd_work_group_size attribute, so we are trying to stick to that in this
    // case.
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<
          xilinx::reqd_work_group_size<4, 4, 4, reqd_work_group_size_test>>(
          k_range, [=](item<3> index) { wb[0] = 1; });
    });

    auto rb = ob.get_access<access::mode::read>();

    if (q.get_context().get_platform().get_info<info::platform::vendor>()
        =="Xilinx") {
      assert(rb[0] == 0); // should still be 0
    } else {
      assert(rb[0] == 1); // shouldn't hinder intel implementation
    }
  }

  std::cout << "reqd_work_group_size_test2 begin \n";

  {
    // Valid and probably main use case for Xilinx FPGAs that allows the kernel
    // to be optimized.
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.single_task<
          xilinx::reqd_work_group_size<1, 1, 1, reqd_work_group_size_test2>>(
          [=]() { wb[0] = 2; });
    });

    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 2);
  }

  std::cout << "reqd_work_group_size_test3 begin (xilinx fail, intel pass) \n";

  nd_range<3> ndr {range<3>{16,16,16}, range<3>{4,4,4}};

  {
    // An invalid use case reqd_work_group_size is bigger than the global range
    // and the local range
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<
          xilinx::reqd_work_group_size<24, 24, 24, reqd_work_group_size_test3>>(
            ndr, [=](nd_item<3> index) { wb[0] = 3; });
    });

    auto rb = ob.get_access<access::mode::read>();
    if (q.get_context().get_platform().get_info<info::platform::vendor>()
        =="Xilinx") {
      assert(rb[0] == 2); // should still be 2
    } else {
      assert(rb[0] == 3); // shouldn't hinder intel implementation
    }
  }

  std::cout << "reqd_work_group_size_test4 begin \n";

  {
    // Should pick up reqd_work_group_size<4, 4, 4> and apply it and ignore the
    // random template inside of the xilinx namespace. As it's 4, 4, 4 the same
    // as nd_range's local it shouldn't break.
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<xilinx::reqd_work_group_size<
          4, 4, 4, xilinx::conflict_test<30, reqd_work_group_size_test4>>>(
          ndr, [=](nd_item<3> index) { wb[0] = 4; });
    });

    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 4);
  }

  std::cout << "reqd_work_group_size_test5 begin \n";

  {
    // This test should pick up reqd_work_group_size<4, 4, 4> which is a valid
    // size and run. If the secondary reqd_work_group_size is chosen incorrectly
    // this should emit a run-time error.
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<xilinx::reqd_work_group_size<4, 4, 4,
                        xilinx::reqd_work_group_size<5, 5, 5,
                          reqd_work_group_size_test5>>>(
        ndr, [=](nd_item<3> index) { wb[0] = 5; });
    });

    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 5);
  }

  std::cout << "xilinx::reqd_work_group_size<16, 16, 16, int> begin \n";

  {
    // TODO/FIXME?: This is an "valid" (compileable and executeable) use case
    // but should it be legal to name kernels like this using a property?
    // It should be possible to restrict it via constraints on the template
    // property.
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<xilinx::reqd_work_group_size<4, 4, 4, int>>(
          ndr, [=](nd_item<3> index) {
            wb[0] = 6;
          });
    });

    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 6);
  }

  std::cout << "reqd_work_group_size_test6 begin (xilinx fail, intel pass) \n";

  {
    // breaks as you are specifying an incorrect workgroup size. It is larger
    // than the nd_range's maximum local size
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<
          xilinx::reqd_work_group_size<8, 8, 8, reqd_work_group_size_test6>>(
          ndr, [=](nd_item<3> index) {
            wb[0] = 7;
          });
    });

    auto rb = ob.get_access<access::mode::read>();
    if (q.get_context().get_platform().get_info<info::platform::vendor>()
        =="Xilinx") {
      assert(rb[0] == 6); // should still be 6
    } else {
      assert(rb[0] == 7); // shouldn't hinder intel implementation
    }
  }

  std::cout << "reqd_work_group_size_test7 begin (xilinx fail, intel pass) \n";

  {
    /*
    Invalid and breaks. While it is smaller than the specified global and
    local_work_size of the nd_range it's not the same as the nd_range...

    We don't want to enforce a reqd_work_group_size of 2x2x2 on something that's
    already defined by an nd_range to be 4x4x4. It's possible to
    enforce the reqd_work_group_size property over the nd_range in the runtime
    but you are breaking the API as the nd_item's properties would be what the
    user specified via the nd_range.
    */
    q.submit([&](handler &cgh) {
    auto wb = ob.get_access<access::mode::write>(cgh);
    cgh.parallel_for<
        xilinx::reqd_work_group_size<2, 2, 2, reqd_work_group_size_test7>>(
        ndr, [=](nd_item<3> index) {
          wb[0] = 8;
        });
    });

    auto rb = ob.get_access<access::mode::read>();
    if (q.get_context().get_platform().get_info<info::platform::vendor>()
        =="Xilinx") {
      assert(rb[0] == 6); // should still be 6
    } else {
      assert(rb[0] == 8); // shouldn't hinder intel implementation
    }
  }

  std::cout << "reqd_work_group_size_test8 begin \n";

  buffer<unsigned int> ob2(range<1>{16*16*16});
  {
    // legal and the one test in here that is intended to give an "interesting"
    // output result for the moment.
    q.submit([&](handler &cgh) {
      auto wb = ob2.get_access<access::mode::write>(cgh);
      cgh.parallel_for<
          xilinx::reqd_work_group_size<4, 4, 4, reqd_work_group_size_test8>>(
          ndr, [=](nd_item<3> index) {
            wb[index.get_global_linear_id()] = index.get_local_linear_id();
          });
    });

    auto rb = ob2.get_access<access::mode::read>();
    unsigned int sum = 0;
    for (size_t i = 0; i != ob2.size(); ++i) {
      sum += rb[i];
    }
    assert(sum == 129024);
  }

  std::cout << "reqd_work_group_size_test9 begin \n";

  {
    // This is perhaps the one time using a reqd_work_group_size on a
    // parallel_for with no local range is legal as it's basically defining
    // a single_task
    range<1> single_range{1};
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.parallel_for<
          xilinx::reqd_work_group_size<1, 1, 1, reqd_work_group_size_test9>>(
          single_range, [=](item<1> index) {
            wb[0] = 9;
          });
    });

    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 9);
  }

  std::cout << "reqd_work_group_size_test10 begin \n";

  {
    // We force this into a reqd_work_group_size by appending the template
    // to the kernel name inside of the handler anyway. So this should be the
    // same as if a user explicitlyapplied the property to the name.
    q.submit([&](handler &cgh) {
      auto wb = ob.get_access<access::mode::write>(cgh);
      cgh.single_task<reqd_work_group_size_test10>(
          [=]() { wb[0] = 10; });
    });

    auto rb = ob.get_access<access::mode::read>();
    assert(rb[0] == 10);
  }

  std::cout << "exiting test \n";

  return 0;
}
