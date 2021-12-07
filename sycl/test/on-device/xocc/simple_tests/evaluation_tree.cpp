// REQUIRES: xocc

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-unnamed-lambda -std=c++20 -Xsycl-target-frontend -fno-exceptions %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <optional>
#include <type_traits>
#include <variant>
namespace detail {

template <typename> struct variant_trait {};

template <typename... Tys> struct variant_trait<std::variant<Tys...>> {
  using indexes = std::make_index_sequence<sizeof...(Tys)>;
};

// // This could be a replacement for visit_single_impl but causes address space casts
// template <typename ret_type, typename Func, typename Var, auto... Idx>
// inline ret_type
// visit_single_impl(Func &&f, std::integer_sequence<size_t, Idx...>, Var &&var) {
//   if constexpr (std::is_same<void, ret_type>::value) {
//     (
//         [&] {
//           if (var.index() == Idx)
//             std::forward<Func>(f)(std::get<Idx>(var));
//         }(),
//         ...);
//   } else {
//     // we use optional here because ret_type may not
//     // be default constructible.
//     std::optional<ret_type> ret;
//     (
//         [&] {
//           if (var.index() == Idx)
//             ret.emplace(std::forward<Func>(f)(std::get<Idx>(var)));
//         }(),
//         ...);
//     return std::forward<ret_type>(*ret);
//   }
// }

template <typename ret_type, typename Func, typename Var>
[[noreturn]] inline ret_type
visit_single_impl(Func&&, std::integer_sequence<size_t>, Var&&) {
  /// assert is a noop on device
  assert(false && "unreachable");
  __builtin_unreachable();
}

template <typename ret_type, typename Func, typename Var, auto First,
          auto... Idx>
inline ret_type visit_single_impl(Func&& f,
                                  std::integer_sequence<size_t, First, Idx...>,
                                  Var&& var) {
  if (var.index() == First)
    return std::forward<Func>(f)(std::get<First>(var));
  return visit_single_impl<ret_type>(std::forward<Func>(f),
                                     std::integer_sequence<size_t, Idx...> {},
                                     std::forward<Var>(var));
}

template <typename Func, typename Var>
decltype(auto) visit_single(Func&& f, Var&& var) {
  assert((!var.valueless_by_exception()));
  using ret_type =
      std::invoke_result_t<Func, decltype(std::get<0>(std::declval<Var>()))>;
  return visit_single_impl<ret_type>(
      std::forward<Func>(f),
      typename variant_trait<
          std::remove_cv_t<std::remove_reference_t<Var>>>::indexes {},
      std::forward<Var>(var));
}
} // namespace detail

/// dev_visit is std::visit implementation suitable to be used in device code.
/// This version of visit doesn't use any function pointer but uses if series
/// which will be turned into switch case by the optimizer.
template <typename Func, typename Var, typename... Rest>
auto dev_visit(Func&& f, Var&& var, Rest&&... rest) {
  if constexpr (sizeof...(Rest) == 0)
    return detail::visit_single(std::forward<Func>(f), std::forward<Var>(var));
  else
    return detail::visit_single(
        [&](auto&& First) {
          return dev_visit(
              [&](auto&&... Others) {
                std::forward<Func>(f)(
                    std::forward<decltype(First)>(First),
                    std::forward<decltype(Others)>(Others)...);
              },
              std::forward<Rest>(rest)...);
        },
        std::forward<Var>(var));
}

/// This is a relative pointer behaving mostly like T*. It doesn't manage the
/// lifetime of the data pointed to. Also this class have shallow const semantic
/// meaning that const rel_ptr<T> is equivalent to T* const not const T *. This
/// pointer should behave like T* for any operation except memcpy or bitcasting
/// the representation of the pointer. When memcpy'ed to another place the
/// offset between the pointer's address and the data it is pointed to will stay
/// constant. This means that, if a pointer and its pointed data are copied with
/// the same memcpy, the pointer is still pointing to the same data. Storing
/// data to files by reinterpreting it in bytes and reading it by reinterpreting
/// bytes into structured data is kind of equivalent to a memcpy across address
/// space. If you simply mmap the file into memory you can use rel_ptr to follow
/// references between different parts of the file.
template <typename T, typename IntTy = std::ptrdiff_t> class rel_ptr {
  IntTy offset;

  char* get_this() const {
    return reinterpret_cast<char*>(const_cast<rel_ptr*>(this));
  }
  IntTy get_offset(T *t) const {
    return (reinterpret_cast<char *>(t) - get_this());
  }

public:
  T *get() const { return (T *)(get_this() + offset); }
  T &operator*() const { return *get(); }
  T *operator->() const { return get(); }
  T &operator[](std::size_t e) const { return get()[e]; }
  explicit operator bool() const { return get(); }
  bool operator!() const { return !get(); }

  rel_ptr() : rel_ptr(nullptr) {}
  explicit rel_ptr(T *t) : offset(get_offset(t)) {}
  rel_ptr(const rel_ptr &other) : rel_ptr(other.get()) {}
  rel_ptr &operator=(const rel_ptr &other) {
    offset = get_offset(other.get());
    return *this;
  }

  friend rel_ptr operator+(const rel_ptr &ptr, IntTy off) {
    return rel_ptr(ptr.get() + off);
  }
  friend rel_ptr operator-(const rel_ptr &ptr, IntTy off) {
    return rel_ptr(ptr.get() - off);
  }
  IntTy operator-(const rel_ptr &other) const { return get() - other.get(); }

  bool operator==(const rel_ptr &other) const { return get() == other.get(); }
  friend bool operator==(const rel_ptr &p1, T *p2) { return p1.get() == p2; }

  /// We use weak_ordering because the bitcasted representation of equal pointer
  /// are different. 2 pointers are equal if they point to the same
  /// object and if they point to the same object, they must have have different
  /// offsets to it, so their representations are different.
  std::weak_ordering operator<=>(const rel_ptr &other) {
    return get() <=> other.get();
  }
  std::weak_ordering operator<=>(T *p2) { return get() <=> p2; }

  rel_ptr &operator++() {
    *this = (*this) + 1;
    return *this;
  }
  rel_ptr operator++(int) {
    rel_ptr tmp = *this;
    ++*this;
    return tmp;
  }
  rel_ptr &operator--() {
    *this = (*this) - 1;
    return *this;
  }
  rel_ptr operator--(int) {
    rel_ptr tmp = *this;
    --*this;
    return tmp;
  }
};

struct Constant {
  int d;
  int compute() { return d; }
};

struct node;

struct BinOp {
  rel_ptr<node> elem[2];
};
struct AddOp : BinOp {
  int compute();
};
struct MulOp : BinOp {
  int compute();
};

struct node {
  std::variant<Constant, AddOp, MulOp> n;
  int cache;
  node(auto v) : n(v) {}
  void compute() {
    /// dev_visit is our own implementation of std::visit that can be used in
    /// kernel code.
    cache = dev_visit([&](auto &e) { return e.compute(); }, n);
  }
  int get_value() { return cache; }
};

int AddOp::compute() {
  return BinOp::elem[0]->get_value() + BinOp::elem[1]->get_value();
}

int MulOp::compute() {
  return BinOp::elem[0]->get_value() * BinOp::elem[1]->get_value();
}

int main() {
  // Computation graph for (1 + 2) * 3
  std::array<node, 5> data{
      {Constant{1}, Constant{2}, AddOp{rel_ptr{&data[0]}, rel_ptr{&data[1]}},
       Constant{3}, MulOp{rel_ptr{&data[2]}, rel_ptr{&data[3]}}}};
  sycl::buffer<node> In{data.data(), {data.size()}};
  sycl::buffer<int> Out{1};

  sycl::queue Queue;

  Queue.submit([&](sycl::handler &cgh) {
    auto AIn = In.get_access<sycl::access::mode::read_write>(cgh);
    auto AOut = Out.get_access<sycl::access::mode::write>(cgh);
    int root_node = data.size() - 1;
    cgh.single_task<class Kernel>([=] {
      for (int i = 0; i < AIn.size(); i++)
        AIn[i].compute();
      AOut[0] = AIn[root_node].get_value();
    });
  });
  {
    auto AOut = Out.get_access<sycl::access::mode::read>();
    std::cout << AOut[0] << std::endl;
    assert(AOut[0] == 9);
    }
  }
