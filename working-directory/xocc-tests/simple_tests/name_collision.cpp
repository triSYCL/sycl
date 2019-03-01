#include <CL/sycl.hpp>

using namespace cl::sycl;

class add_2;

// Perhaps a question to ask in the Khronos group for clarity? Are either of these legal?
// Is it specified somewhere that it isn't legal?
int main() {
  queue q;

  q.submit([&] (handler &cgh) {
    cgh.single_task<add_2>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<add_2>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<class add>([=] () {
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.single_task<class add>([=] () {
    });
  });

  return 0;
}
