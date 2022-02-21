// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out -Xsycl-target-frontend -fno-exceptions
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <CL/sycl.hpp>
#include <variant>

#include "../utilities/device_selectors.hpp"

namespace detail {

template<typename> struct variant_trait {};

template<typename... Tys> struct variant_trait<std::variant<Tys...>> {
  using indexes = std::make_index_sequence<sizeof...(Tys)>;
};

template<typename ret_type, typename Func, typename Var>
[[noreturn]] inline ret_type
visit_single_impl(Func &&, std::integer_sequence<size_t>, Var &&) {
  __builtin_unreachable();
}

template<typename ret_type, typename Func, typename Var, auto First,
         auto... Idx>
inline ret_type visit_single_impl(Func &&f,
                                  std::integer_sequence<size_t, First, Idx...>,
                                  Var &&var) {
  if (var.index() == First)
    return std::forward<Func>(f)(std::get<First>(var));
  return visit_single_impl<ret_type>(std::forward<Func>(f),
                                     std::integer_sequence<size_t, Idx...>{},
                                     std::forward<Var>(var));
}

template<typename Func, typename Var>
decltype(auto) visit_single(Func &&f, Var &&var) {
  assert((!var.valueless_by_exception()));
  using ret_type =
      std::invoke_result_t<Func, decltype(std::get<0>(std::declval<Var>()))>;
  return visit_single_impl<ret_type>(
      std::forward<Func>(f),
      typename variant_trait<
          std::remove_cv_t<std::remove_reference_t<Var>>>::indexes{},
      std::forward<Var>(var));
}
} // namespace detail

/// implementations of std::visit is libstdc++ and libc++ use function pointers
/// because of this they can't be used in device code.
/// This is a implementation of std::visit that is suitable for device code.
template <typename Func, typename Var, typename... Rest>
auto dev_visit(Func &&f, Var &&var, Rest &&...rest) {
  if constexpr (sizeof...(Rest) == 0)
    return detail::visit_single(std::forward<Func>(f), std::forward<Var>(var));
  else
    return detail::visit_single(
        [&](auto &&First) {
          return dev_visit(
              [&](auto &&...Others) {
                std::forward<Func>(f)(
                    std::forward<decltype(First)>(First),
                    std::forward<decltype(Others)>(Others)...);
              },
              std::forward<Rest>(rest)...);
        },
        std::forward<Var>(var));
}

struct A {
  float f;
  int i;
};

struct B {
  int i;
  float f;
};

int main() {
  std::vector<std::variant<A, B>> d;
  d.push_back(A{1.1, 2});
  d.push_back(B{1, 2.1});
  sycl::buffer<std::variant<A, B>> In{d.data(), {d.size()}};
  sycl::buffer<std::variant<B, A>> Out{d.size()};
  sycl::buffer<int> OutI{1};

  sycl::queue Queue{ selector_defines::CompiledForDeviceSelector {} };

  Queue.submit([&](sycl::handler &cgh) {
    auto AIn = In.get_access<sycl::access::mode::read>(cgh);
    auto AOut = Out.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class Kernel>([=] {
      for (unsigned i = 0; i < AIn.size(); i++)
        dev_visit([&](auto V) { AOut[{i}] = V; }, AIn[{i}]);
    });
  });
  {
    auto AOut = Out.get_access<sycl::access::mode::read>();
    dev_visit([&](auto V) { assert(V.i == 2 && (int)V.f == 1); }, AOut[{0}]);
    dev_visit([&](auto V) { assert(V.i == 1 && (int)V.f == 2); }, AOut[{1}]);
  }
}
