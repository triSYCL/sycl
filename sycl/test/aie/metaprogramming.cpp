// REQUIRES: aie

// RUN: %aie_clang %s -o %t.bin

#include "aie.hpp"

int main() {
  struct A {
    struct data_type {};
    static uint32_t act_on_data(bool x, int y, int, data_type) { return 0; }
  };
  struct B {
    struct data_type {};
    static uint32_t act_on_data(bool x, int y, int, data_type) { return 0; }
  };
  struct C {
    struct data_type {};
    static bool act_on_data(bool x, int y, int, data_type) { return 0; }
  };

  struct D {};

  using tseq = aie::detail::type_seq<A, B, C>;

  static_assert(std::is_same_v<aie::detail::func_info_t<A::act_on_data>::ret_type, uint32_t>, "");
  static_assert(std::is_same_v<aie::detail::func_info_t<A::act_on_data>::args, aie::detail::type_seq<bool, int, int, A::data_type>>, "");
  static_assert(std::is_same_v<aie::detail::func_info_t<A::act_on_data>::args::get_type<0>, bool>, "");
  static_assert(aie::detail::func_info_t<A::act_on_data>::args::size == 4, "");

  static_assert(std::is_same_v<tseq::get_type<0>, A>, "");
  static_assert(std::is_same_v<tseq::get_type<1>, B>, "");
  static_assert(std::is_same_v<tseq::get_type<2>, C>, "");

  static_assert(tseq::get_index<A> == 0, "");
  static_assert(tseq::get_index<B> == 1, "");
  static_assert(tseq::get_index<C> == 2, "");

  using rpc = aie::detail::rpcs_info<A, B, C>;

  static_assert(std::is_same_v<rpc::data_seq, aie::detail::type_seq<A::data_type, B::data_type, C::data_type>>, "");
  static_assert(std::is_same_v<rpc::ret_seq, aie::detail::type_seq<uint32_t, uint32_t, bool>>, "");

  rpc::for_any(0, [&]<typename T>(){
    static_assert(std::is_same_v<T, A> || std::is_same_v<T, B> || std::is_same_v<T, C>, "");
    assert((std::is_same_v<T, A>));
  });
  rpc::for_any(1, [&]<typename T>(){
    static_assert(std::is_same_v<T, A> || std::is_same_v<T, B> || std::is_same_v<T, C>, "");
    assert((std::is_same_v<T, B>));
  });
  rpc::for_any(2, [&]<typename T>(){
    static_assert(std::is_same_v<T, A> || std::is_same_v<T, B> || std::is_same_v<T, C>, "");
    assert((std::is_same_v<T, C>));
  });
}
