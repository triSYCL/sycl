#include <CL/sycl.hpp>

#include "../utilities/device_selectors.hpp"

using namespace cl::sycl;

class math_mangle;

int main(int argc, char* argv[]) {
  selector_defines::XOCLDeviceSelector xocl;

  queue q { xocl };
  q.submit([&](handler &cgh) {
    cgh.single_task<math_mangle>([=]() {
      printf("in single_task \n");
      printf("sqrt: %f \n", cl::sycl::sqrt(10.0f));
      printf("fabs: %f \n", cl::sycl::fabs(-10.0f));
      printf("sin: %f \n", cl::sycl::sin(10.0f));
      printf("max: %d \n", cl::sycl::max(10, 12));
      printf("min: %d \n", cl::sycl::min(10, 12));
      printf("floor: %f \n", cl::sycl::floor(10.2f));
      printf("ceil: %f \n", cl::sycl::ceil(10.2f));
      printf("exp: %f \n", cl::sycl::exp(10.2f));
      printf("log: %f \n", cl::sycl::log(10.2f));
      printf("fmin: %f \n", cl::sycl::fmin(10.2f, 10.0f));
      printf("fmax: %f \n", cl::sycl::fmax(10.2f, 10.0f));
      printf("exp: %f \n", cl::sycl::exp(10.2f));
      printf("cos: %f \n", cl::sycl::cos(10.2f));
      // printf("mad: %f \n", cl::sycl::mad(10.2f, 11.0f, 12.0f)); // not working, unsure why at the moment
    });
  });

  q.wait();

  return 0;
}
