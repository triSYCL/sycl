//==- cstr.hpp --- SYCL utilities to manipulate string constants     -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  This file contains meta programming utilities to manipulate string
///  constants stored in a type like cstr<char, 'a', 'b'>.
///  This is used for kernel properties because they must be constant and the
///  API reflects that.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_DETAIL_CSTR_HPP
#define SYCL_DETAIL_CSTR_HPP

#include <algorithm>

#include "CL/sycl/detail/pi.h"

__SYCL_INLINE_NAMESPACE(cl) {

namespace sycl {

namespace detail {

#ifdef __clang__
template <std::size_t I, typename... Ts>
using type_at = __type_pack_element<I, Ts...>;
#else
template <unsigned I, typename T, typename... Ts> struct type_at_impl {
  using type = typename type_at_impl<I - 1, Ts...>::type;
};

template <typename T, typename... Ts> struct type_at_impl<0, T, Ts...> {
  using type = T;
};

template <unsigned I, typename... Ts>
using type_at = typename type_at_impl<I, Ts...>::type;
#endif

/// Transform a type like: cstr<char, 'a', 'b'>
/// into a string "ab"
template <typename> struct StrPacker {};

/// Utility type to manipulate string in constant context
template <typename CharT, CharT... charpack> struct cstr {
  using Char = CharT;
  static constexpr unsigned size = sizeof...(charpack);
  /// Access character at position Idx in the string
  template <unsigned Idx> using at = type_at<Idx, cstr<CharT, charpack>...>;
};

template <typename CharT> struct cstr<CharT> {
  using Char = CharT;
  static constexpr unsigned size = 0;
};

template <typename CharT, CharT... P1> struct StrPacker<cstr<CharT, P1...>> {
  static constexpr const CharT Str[] = {P1...};
};

/// Utility to merge string with a separator
template <typename...> struct concat {};

template <typename CharT, CharT... Sep, CharT... P1>
struct concat<cstr<CharT, Sep...>, cstr<CharT, P1...>> {
  using type = cstr<CharT, P1...>;
};

template <typename CharT, CharT... Sep, CharT... P1, CharT... P2,
          typename... Rest>
struct concat<cstr<CharT, Sep...>, cstr<CharT, P1...>, cstr<CharT, P2...>,
              Rest...> {
  using type =
      typename concat<cstr<CharT, Sep...>, cstr<CharT, P1..., Sep..., P2...>,
                      Rest...>::type;
};

template <typename... Ts> using concat_t = typename concat<Ts...>::type;

inline namespace literals {
/// We are using a compiler extention here and we don't want any warning about
/// it so we suppress that warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-string-literal-operator-template"

/// Utility to transform a string literal into its cstr form.
template <typename CharT, CharT... charpack>
__SYCL_ALWAYS_INLINE constexpr cstr<CharT, charpack...>
operator"" _cstr() noexcept {
  return {};
}
#pragma clang diagnostic pop
} // namespace literals

/// Define some bases for number manipulation.
/// Base*::size is used to get the size of the base(2, 10, 16).
/// Base*::at<N> is used to get the didgit that represent the value N.
using Base2 = decltype("01"_cstr);
using Base10 = decltype("0123456789"_cstr);
using Base16 = decltype("0123456789abcdef"_cstr);

#ifdef __SYCL_DEVICE_ONLY__
/// Utility to transform a number constant like 348 into
/// cstr<char, '3', '4', '8'> such that it can be manipulated as a string.
template <bool, typename> struct StrNumber {};
#else
template <bool, typename> struct StrNumber { using type = cstr<char>; };
#endif

template <auto N, typename Base = Base10> struct number {
  static constexpr auto value = N;
  static constexpr auto str =
      typename StrNumber<(N >= Base::size), number>::type{};
};

#ifdef __SYCL_DEVICE_ONLY__
template <auto N, typename Base> struct StrNumber<true, number<N, Base>> {
  using last_didgit = typename Base::template at<N % Base::size>;
  using type = concat_t<cstr<char>,
                        typename StrNumber<(N / Base::size >= Base::size),
                                           number<N / Base::size, Base>>::type,
                        last_didgit>;
};

template <auto N, typename Base> struct StrNumber<false, number<N, Base>> {
  using type = typename Base::template at<N % Base::size>;
};
#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif
