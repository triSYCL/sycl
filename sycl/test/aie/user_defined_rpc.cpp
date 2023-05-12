// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"

int counter = 0;

struct counter_rpc {
  struct data_type {
  };
  template<typename Parent>
  struct add_to_dev_handle {
    private:
    auto* get() { return static_cast<Parent*>(this)->dt(); }
    public:
    uint32_t get_counter() {
      data_type data{};
      return get()->perform_rpc(data);
    }
  };
  static uint32_t act_on_data(int x, int y, aie::device_mem_handle h, data_type d) {
    return counter++;
  }
};

struct nothing_rpc {
  struct data_type {
  };
  template<typename Parent>
  struct add_to_dev_handle {
    private:
    auto* get() { return static_cast<Parent*>(this)->dt(); }
    public:
    void nothing() {
      data_type data{};
      return get()->perform_rpc(data);
    }
  };
  static void act_on_data(int x, int y, aie::device_mem_handle h, data_type d) {
  }
};

int main() {
  aie::device<1, 1> dev;
  aie::queue q(dev);
  q.submit([](auto &ht) {
    ht.single_task([](auto &dt) {
      dt.rpcs().log("count: ", dt.rpcs().get_counter());
// CHECK: count: 0
      dt.rpcs().log("count: ", dt.rpcs().get_counter());
// CHECK: count: 1
      dt.rpcs().log("count: ", dt.rpcs().get_counter());
// CHECK: count: 2
    });
  }, aie::add_rpc<nothing_rpc, counter_rpc>());
}
// CHECK: exit_code=0
