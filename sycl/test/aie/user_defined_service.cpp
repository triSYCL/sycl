// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int counter = 0;

struct counter_service {
  struct data_type {};

  /// I am not sure it is really simpler by using add_to_api_base than without
  template <typename Parent>
  struct add_to_service_api : aie::add_to_api_base<add_to_service_api<Parent>> {
    using base = aie::add_to_api_base<add_to_service_api<Parent>>;
    uint32_t get_counter() {
      data_type data{};
      return base::tile().perform_service(data);
    }
  };
  uint32_t act_on_data(int x, int y, aie::device_mem_handle h,
                              data_type d) {
    return counter++;
  }
};

struct nothing_service {
  struct data_type {};
  template <typename Parent> struct add_to_service_api {
  private:
    auto& tile() { return *static_cast<Parent*>(this)->dt(); }

  public:
    void nothing() {
      data_type data{};
      return tile().perform_service(data);
    }
  };
  void act_on_data(int x, int y, aie::device_mem_handle h, data_type d) {
  }
};

int main() {
  aie::device<1, 1> dev;
  aie::queue q(dev);
  q.submit(
      [](auto& ht) {
        ht.single_task([](auto& dt) {
          dt.service().log("count: ", dt.service().get_counter());
          // CHECK: count: 0
          dt.service().log("count: ", dt.service().get_counter());
          // CHECK: count: 1
          dt.service().log("count: ", dt.service().get_counter());
          // CHECK: count: 2
        });
      },
      aie::add_service<nothing_service, counter_service>());
}
// CHECK: exit_code=0
