//==---------- reproducer.cpp - XRT Plugin ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header is not quite self contained because it needs to know about xrt objects
// This could be refactored into something generic

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace detail {
template <typename T> constexpr const auto &RawTypeName() {
#ifdef _MSC_VER
  return __FUNCSIG__;
#else
  return __PRETTY_FUNCTION__;
#endif
}

struct RawTypeNameFormat {
  std::size_t leading_junk = 0, trailing_junk = 0;
};

// Returns `false` on failure.
inline constexpr bool GetRawTypeNameFormat(RawTypeNameFormat *format) {
  const auto &str = RawTypeName<int>();
  for (std::size_t i = 0;; i++) {
    if (str[i] == 'i' && str[i + 1] == 'n' && str[i + 2] == 't') {
      if (format) {
        format->leading_junk = i;
        format->trailing_junk =
            sizeof(str) - i - 3 - 1; // `3` is the length of "int", `1` is the
                                     // space for the null terminator.
      }
      return true;
    }
  }
  return false;
}

inline static constexpr RawTypeNameFormat format = [] {
  static_assert(
      GetRawTypeNameFormat(nullptr),
      "Unable to figure out how to generate type names on this compiler.");
  RawTypeNameFormat format;
  GetRawTypeNameFormat(&format);
  return format;
}();
} // namespace detail

// Returns the type name in a `std::array<char, N>` (null-terminated).
template <typename T> [[nodiscard]] constexpr auto CexprTypeName() {
  constexpr std::size_t len = sizeof(detail::RawTypeName<T>()) -
                              detail::format.leading_junk -
                              detail::format.trailing_junk;
  std::array<char, len> name{};
  for (std::size_t i = 0; i < len - 1; i++)
    name[i] = detail::RawTypeName<T>()[i + detail::format.leading_junk];
  return name;
}

template <typename T> [[nodiscard]] const char *TypeName() {
  static constexpr auto name = CexprTypeName<T>();
  return name.data();
}
template <typename T> [[nodiscard]] const char *TypeName(const T &) {
  return TypeName<T>();
}

constexpr const char *reproducer_env_name = "SYCL_PI_XRT_REPRODUCER_PATH";

namespace detail {

std::unordered_map<const void *, std::string> &name_map() {
  static std::unordered_map<const void *, std::string> name_map;
  return name_map;
}

std::string nameof(const void *ptr, bool is_use = true,
                   const std::string &format = "%s") {
  static int id_next = 0;
  if (!std::getenv(reproducer_env_name))
    return "";
  auto &name = name_map()[ptr];
  if (!is_use) {
    std::string id = "name" + std::to_string(id_next++);
    int e = std::snprintf(nullptr, 0, format.c_str(), id.c_str());
    name.resize(e + 1);
    std::snprintf(&name[0], name.size(), format.c_str(), id.c_str());
    name.resize(e);
    return id;
  }
  assert(!name.empty());
  return name;
}

inline bool is_named(const void *ptr) {
  return name_map().find(ptr) != name_map().end();
}

}

std::ostream &reproducer() {
  static std::ofstream of = [] {
    std::ofstream of;
    if (const char *path = std::getenv(reproducer_env_name)) {
      of.open(path);
    }
    return of;
  }();
  return of;
}

namespace detail {

template <typename T,
          typename std::enable_if_t<
              std::is_same_v<void, std::void_t<decltype(reproducer()
                                                        << std::declval<T>())>>,
              int> = 0>
constexpr std::true_type is_streamable_impl(T) {
  return {};
}
constexpr std::false_type is_streamable_impl(...) { return {}; }

template <typename T,
          typename std::enable_if_t<
              std::is_same_v<
                  void, std::void_t<decltype(std::declval<T>().get_handle())>>,
              int> = 0>
constexpr std::true_type has_get_handle_impl(T) {
  return {};
}
constexpr std::false_type has_get_handle_impl(...) { return {}; }

template <typename T>
using remove_cvref_t = std::remove_reference_t<std::remove_cv_t<T>>;

template <typename T>
constexpr bool is_streamable =
    decltype(is_streamable_impl(std::declval<T>()))::value;

template <typename T>
constexpr bool has_get_handle =
    decltype(has_get_handle_impl(std::declval<T>()))::value;

template <typename T>
constexpr bool is_string_like =
    std::is_same_v<std::string, remove_cvref_t<T>> ||
    std::is_same_v<const char *, remove_cvref_t<T>>;

template <typename T>
constexpr bool is_ptr_obj_like =
    std::is_pointer_v<remove_cvref_t<T>> && !is_string_like<T>;

template <typename T>
constexpr bool is_namable = is_ptr_obj_like<T> || has_get_handle<T> ||
                            std::is_same_v<remove_cvref_t<T>, xrt::uuid>;

template <typename T> const void *get_id(const T &t) {
  static std::set<std::string> str_set;
  static_assert(is_namable<T>);
  if constexpr (is_ptr_obj_like<T>) {
    return t;
  } else if constexpr (has_get_handle<T>) {
    return t.get_handle().get();
  } else {
    auto ptr = str_set.find(t.to_string());
    return &*ptr;
  }
}

template <typename T, typename std::enable_if_t<
                          std::is_integral_v<T> || std::is_enum_v<T>, int> = 0>
void print_arg(const T &val) {
  reproducer() << "((" << TypeName<T>() << ")" << val << ")";
}

template <typename T, typename std::enable_if_t<is_string_like<T>, int> = 0>
void print_arg(const T &val) {
  reproducer() << "\"" << val << "\"";
}

template <typename T, typename std::enable_if_t<is_namable<T>, int> = 0>
void print_arg(const T &val) {
  if constexpr (is_ptr_obj_like<T>) {
    if (!is_named(get_id(val))) {
      /// TODO: Should be minimized or removed
      reproducer() << "/*TODO*/" << val;
      return;
    }
  }
  reproducer() << nameof(get_id(val));
}

void print_args(int) {}

template <typename T, typename... Ts>
void print_args(int i, const T &t, const Ts &...ts) {
  if (i != 0)
    reproducer() << ", ";
  print_arg(t);
  print_args(i + 1, ts...);
}

template <typename CallTy, typename... Ts>
auto reproducer_call_wrapper(const std::string &from,
                             const std::string &call_str,
                             const std::string &call_cpp,
                             std::initializer_list<std::string> args_str,
                             CallTy &&call, Ts &&...ts) {
  reproducer() << "// from: " << from << "\n";
  reproducer() << "// call str:" << call_str << "(";
  bool is_first = true;
  for (const std::string &str : args_str) {
    if (!is_first)
      reproducer() << ", ";
    reproducer() << str;
    is_first = false;
  }
  reproducer() << ")\n";
  using ret_type =
      decltype(std::forward<CallTy>(call)(std::forward<Ts>(ts)...));
  auto simple_run = [&] {
    reproducer() << call_cpp << "(";
    print_args(0, ts...);
    reproducer() << ");\n\n";
    std::flush(reproducer());
    return std::forward<CallTy>(call)(std::forward<Ts>(ts)...);
  };
  /// NOT the same as A || B because B cannot be instantiated for void
  if constexpr (std::is_same_v<void, ret_type>)
    return simple_run();
  else if constexpr (!is_namable<ret_type>)
    return simple_run();
  else {
    auto ret = std::forward<CallTy>(call)(std::forward<Ts>(ts)...);
    reproducer() << "auto " << nameof(get_id(ret), false) << " = " << call_cpp
                 << "(";
    print_args(0, ts...);
    reproducer() << ");\n\n";
    std::flush(reproducer());
    return std::forward<ret_type>(ret);
  }
}

template <typename ObjTy, typename CallTy, typename... Ts>
auto reproducer_memcall_unpacker(const std::string &from,
                                 const std::string &obj_str,
                                 const std::string &call_str,
                                 std::initializer_list<std::string> args_str,
                                 const ObjTy &obj, CallTy &&call,
                                 std::tuple<Ts...> ts) {
  return std::apply(
      [&](auto &&...args) {
        return reproducer_call_wrapper(
            from, obj_str + "." + call_str,
            nameof(obj.get_handle().get()) + "." + call_str, args_str,
            std::forward<CallTy>(call), std::forward<decltype(args)>(args)...);
      },
      ts);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename Tuple>
auto reproducer_call_unpacker(T1 &&t1, T2 &&t2, T3 &&t3, T4 &&t4, T5 &&t5,
                              Tuple &&ts) {
  return std::apply(
      [&](auto &&...args) {
        return reproducer_call_wrapper(
            std::forward<T1>(t1), std::forward<T2>(t2), std::forward<T3>(t3),
            std::forward<T4>(t4), std::forward<T5>(t5),
            std::forward<decltype(args)>(args)...);
      },
      ts);
}

const char *print_char(unsigned char c, std::array<char, 10> &out) {
  static char table[] = "0123456789abcdef";
  int i = 0;
  out[i++] = '\'';
  out[i++] = '\\';
  out[i++] = 'x';
  out[i++] = table[c / 16];
  out[i++] = table[c % 16];
  out[i++] = '\'';
  out[i++] = '\0';
  return out.data();
}

template <typename T>
void reproducer_add_buffer_wrapper(const std::string &from, T *ptr, int size) {
  reproducer() << "// from: " << from << "\n";
  reproducer() << "std::array<char, " << size << "> "
               << nameof(ptr, false,
                         std::string("((") + TypeName<T>() + "*)%s.data())")
               << "= {";
  std::array<char, 10> char_buff;
  const unsigned char *cptr = (const unsigned char *)ptr;
  if (size > 0) {
    reproducer() << print_char(cptr[0], char_buff);
    for (int i = 1; i < size; i++) {
      reproducer() << ", " << print_char(cptr[i], char_buff);
    }
  }
  reproducer() << "};\n\n";
  std::flush(reproducer());
}

template <typename T, typename T2>
void reproducer_add_related_ptr_wrapper(T *base, T2* ptr) {
  int offset = ((char*)ptr) - ((char*)base);
  if (offset == 0)
    return;
  assert(offset >= 0);
  nameof(ptr, false, "(void*)(((char*)" + nameof(base) + ") + " + std::to_string(offset) + ")");
}
}

/// allow treating constructors and function call the same way
#define CALLABLE(X)                                                            \
  [&](auto &&...args) { return X(std::forward<decltype(args)>(args)...); }

/// generate: X(...) or auto nameY = X(...)
#define REPRODUCE_CALL(X, ...)                                                 \
  ::detail::reproducer_call_unpacker(                                          \
      std::string(__PRETTY_FUNCTION__) + " " + __FILE__ + ":" +                \
          std::to_string(__LINE__),                                            \
      #X, #X, std::initializer_list<std::string>{#__VA_ARGS__}, CALLABLE(X),   \
      std::forward_as_tuple(__VA_ARGS__))

/// generate: O.X(...) or auto nameY = O.X(...)
#define REPRODUCE_MEMCALL(O, X, ...)                                           \
  ::detail::reproducer_memcall_unpacker(                                       \
      std::string(__PRETTY_FUNCTION__) + " " + __FILE__ + ":" +                \
          std::to_string(__LINE__),                                            \
      #O, #X, std::initializer_list<std::string>{#__VA_ARGS__}, (O),           \
      CALLABLE((O).X), std::forward_as_tuple(__VA_ARGS__))

/// replace P by a buffer of S elements
/// generate: std::array<X, S> = {...};
#define REPRODUCE_ADD_BUFFER(P, S)                                             \
  ::detail::reproducer_add_buffer_wrapper(std::string(__PRETTY_FUNCTION__) +   \
                                              " " + __FILE__ + ":" +           \
                                              std::to_string(__LINE__),        \
                                          (P), (S))

/// Inform the naming system that P should be expressed (B + X)
/// generate nothing on its own.
/// this is a macro only be cause the rest of the API is macros.
#define REPRODUCE_ADD_RELATED_PTR(B, P)                                        \
  ::detail::reproducer_add_related_ptr_wrapper(B, P)
