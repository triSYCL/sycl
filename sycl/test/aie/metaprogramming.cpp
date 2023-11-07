// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin
// Exercise some implementation details
#include "aie.hpp"

struct A {
  struct data_type {};
  template <typename Parent> struct add_to_service_api {};
  uint32_t act_on_data(bool x, int y, aie::device_mem_handle, data_type) {
    return 0;
  }
};
struct B {
  struct data_type {};
  template <typename Parent> struct add_to_service_api {};
  uint32_t act_on_data(bool x, int y, aie::device_mem_handle, data_type) {
    return 0;
  }
};
struct C {
  struct data_type {};
  template <typename Parent> struct add_to_service_api {};
  bool act_on_data(bool x, int y, aie::device_mem_handle, data_type) {
    return 0;
  }
};

int main() {
  struct D {};

  using tseq = aie::detail::type_seq<A, B, C>;

  static_assert(
      std::is_same_v<aie::detail::memfunc_info_t<&A::act_on_data>::ret_type,
                     uint32_t>,
      "");
  static_assert(
      std::is_same_v<aie::detail::memfunc_info_t<&A::act_on_data>::args,
                     aie::detail::type_seq<bool, int, aie::device_mem_handle,
                                           A::data_type>>,
      "");
  static_assert(
      std::is_same_v<
          aie::detail::memfunc_info_t<&A::act_on_data>::args::get_type<0>,
          bool>,
      "");
  static_assert(aie::detail::memfunc_info_t<&A::act_on_data>::args::size == 4,
                "");

  static_assert(std::is_same_v<tseq::get_type<0>, A>, "");
  static_assert(std::is_same_v<tseq::get_type<1>, B>, "");
  static_assert(std::is_same_v<tseq::get_type<2>, C>, "");

  static_assert(tseq::get_index<A> == 0, "");
  static_assert(tseq::get_index<B> == 1, "");
  static_assert(tseq::get_index<C> == 2, "");

  using service = aie::detail::service_list_info<A, B, C>;

  static_assert(
      std::is_same_v<
          service::data_seq,
          aie::detail::type_seq<A::data_type, B::data_type, C::data_type>>,
      "");
  static_assert(std::is_same_v<service::ret_seq,
                               aie::detail::type_seq<uint32_t, uint32_t, bool>>,
                "");

  service::for_any(0, [&]<typename T>() {
    static_assert(std::is_same_v<T, A> || std::is_same_v<T, B> ||
                      std::is_same_v<T, C>,
                  "");
    assert((std::is_same_v<T, A>));
  });
  service::for_any(1, [&]<typename T>() {
    static_assert(std::is_same_v<T, A> || std::is_same_v<T, B> ||
                      std::is_same_v<T, C>,
                  "");
    assert((std::is_same_v<T, B>));
  });
  service::for_any(2, [&]<typename T>() {
    static_assert(std::is_same_v<T, A> || std::is_same_v<T, B> ||
                      std::is_same_v<T, C>,
                  "");
    assert((std::is_same_v<T, C>));
  });
}
