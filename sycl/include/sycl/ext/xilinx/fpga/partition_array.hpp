//==- partition_array.hpp --- SYCL Xilinx array partition extension  -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a class expressing arrays that can be partitioned.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_PARTITION_ARRAY_HPP
#define SYCL_XILINX_FPGA_PARTITION_ARRAY_HPP

#include "sycl/detail/defines.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace ext::xilinx {

/// Type used to represent dimensions of an partition_ndarray
template<std::size_t...>
struct dim {};

/** Kind of array partition

    To be used when defining or declaring a vendor-supported partition
    array in kernel.
*/
namespace partition {
  /** Three real partition types: cyclic, block, complete.

      none represents non partitioned standard array.
  */
  namespace type {
  enum type : int {
    cyclic,
    block,
    complete,
    none
  };
  }

  /// This fuction is currently empty but the LowerSYCLMetaData Pass will fill
  /// it with the required IR.
  template<typename Ptr>
#if defined(__SYCL_XILINX_HW_EMU_MODE__) || defined(__SYCL_XILINX_HW_MODE__)
  __SYCL_DEVICE_ANNOTATE("xilinx_partition_array")
#endif
  __SYCL_ALWAYS_INLINE
  inline void xilinx_partition_array(Ptr, int, int, int) {}

  /// This fuction is currently empty but the LowerSYCLMetaData Pass will fill
  /// it with the required IR.
  template<typename Ptr>
#if defined(__SYCL_XILINX_HW_EMU_MODE__) || defined(__SYCL_XILINX_HW_MODE__)
  __SYCL_DEVICE_ANNOTATE("xilinx_bind_storage")
#endif
  __SYCL_ALWAYS_INLINE
  inline void xilinx_bind_storage(Ptr, int, int, int) {}
  // xilinx_bind_storage(ptr, 666, 18, -1) is a RAM_1P BRAM.

  /** Represent a cyclic partition.

      The single array would be partitioned into several small physical
      memories in this case. These small physical memories can be
      accessed simultaneously which drive the performance. Each
      element in the array would be partitioned to each memory in order
      and cyclically.

      That is if we have a 4-element array which contains 4 integers
      0, 1, 2, and 3. If we set factor to 2, and partition
      dimension to 1 for this cyclic partition array. Then, the
      contents of this array will be distributed to 2 physical
      memories: one contains 1, 3 and the other contains 2,4.

      \param SplitInto is the number of physical memories that user wants to
      have.

      \param PDim is the dimension that user wants to apply cyclic partition on.
      If PDim is 0, all dimensions will be partitioned with cyclic order.
  */
  template <std::size_t SplitInto = 1, std::size_t PDim = 0>
  struct cyclic {
    static constexpr auto split_into = SplitInto;
    static constexpr auto partition_dim = PDim;
    static constexpr auto partition_type = type::cyclic;
  };


  /** Represent a block partition.

      The single array would be partitioned into several small
      physical memories and can be accessed simultaneously, too.
      However, the first physical memory will be filled up first, then
      the next.

      That is if we have a 4-element array which contains 4 integers
      0, 1, 2, and 3. If we set factor to 2, and partition
      dimension to 1 for this cyclic partition array. Then, the
      contents of this array will be distributed to 2 physical
      memories: one contains 1, 2 and the other contains 3,4.

      \param SplitInto is the number blocks the array will be split into.

      \param PDim is the dimension that user wants to apply block partition on.
      If PDim is 0, all dimensions will be partitioned with block order.
  */
  template <std::size_t SplitInto = 1, std::size_t PDim = 0>
  struct block {
    static constexpr auto split_into = SplitInto;
    static constexpr auto partition_dim = PDim;
    static constexpr auto partition_type = type::block;
  };


  /** Represent a complete partition.

      The single array would be partitioned into individual elements.
      That is if we have a 4-element array with one dimension, the
      array is completely partitioned into distributed RAM or 4
      independent registers.

      \param PDim is the dimension that user wants to apply complete partition
      on. If PDim is 0, all dimensions will be completely partitioned.
  */
  template <std::size_t PDim = 0>
  struct complete {
    static constexpr auto partition_dim = PDim;
    static constexpr auto partition_type = type::complete;
  };


  /** Represent a none partition.

      The single array would be the same as std::array.
  */
  struct none {
    static constexpr auto partition_type = type::none;
  };
}  // namespace partition

namespace detail {

template<typename T, auto... Idxs>
struct rec_array_of;

/// Type containing the iterative case of various recursive type and value calculations
template<typename T, auto Idx, auto... Idxs>
struct rec_array_of<T, Idx, Idxs...> {
  using sub_type = rec_array_of<T, Idxs...>;

  using type = typename sub_type::type[Idx];
  using init = std::initializer_list<typename sub_type::init>;

  template<typename InitTy, typename Lambda>
  static inline constexpr bool recursively_on_each(type& self, InitTy&& other, Lambda&& L) {
    auto eIt = std::begin(self);
    auto iIt = std::begin(other);
    for (; eIt < std::end(self) && iIt < std::end(other);eIt++, iIt++)
      if (sub_type::recursively_on_each(*eIt, *iIt, L))
        return true;
    return false;
  }
};

/// Type containing the base case of various recursive type and value calculations
template<typename T>
struct rec_array_of<T> {
  using type = T;
  using init = T;

  template <typename Lambda,
            typename std::enable_if_t<
                std::is_same_v<decltype(std::declval<Lambda>()(
                                   std::declval<type &>(),
                                   std::declval<const init &>())),
                               bool>,
                int> = 0>
  static inline constexpr bool recursively_on_each(type &self, const init &other, Lambda &&L) {
    return L(self, other);
  }
  template <typename Lambda,
            typename std::enable_if_t<
                !std::is_same_v<decltype(std::declval<Lambda>()(
                                    std::declval<type &>(),
                                    std::declval<const init &>())),
                                bool>,
                int> = 0>
  static inline constexpr bool recursively_on_each(type &self, const init &other, Lambda &&L) {
    L(self, other);
    return false;
  }
};

}  // namespace detail

/** Define an array class with partition feature.

    Since on FPGA, users can customize the memory architecture in the system
    and within the CU. Array partition help us to partition single array to
    multiple memories that can be accessed simultaneously getting higher memory
    bandwidth.

    \param ValueType is the type of element.

    \param Size is the size of the array.

    \param PartitionType is the array partition type: cyclic, block, and
    complete. The default type is none.
*/
template <typename ValueType,
          typename Size,
          typename PartitionType = partition::none>
struct partition_ndarray {};

/// alias declaration to make a partition_array = partition_ndarray of 1 dimension
template<typename VTy, std::size_t Size, typename PartitionType = partition::none>
using partition_array = partition_ndarray<VTy, dim<Size>, PartitionType>;

/// The implementation of the N-dimension array
template <typename ValueType, std::size_t Size, std::size_t... Sizes,
          typename PartitionType>
class partition_ndarray<ValueType, dim<Size, Sizes...>, PartitionType> {
  using recursive_type = detail::rec_array_of<ValueType, Size, Sizes...>;

  /// The N-dimension array. 
  typename recursive_type::type elems;

  public:
  /// The kind of partitioning
  static constexpr auto partition_type = PartitionType::partition_type;

  /// Type of the last dimension's element
  using elem_type = ValueType;
  /// Type of the next dimension's element
  using value_type = std::remove_cvref_t<decltype(elems[0])>;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  using dim_type = dim<Size, Sizes...>;

  /// Provide iterator
  iterator begin() { return std::begin(elems); }
  const_iterator begin() const { return std::begin(elems); }
  iterator end() { return std::end(elems); }
  const_iterator end() const { return std::end(elems); }

  /// Evaluate size
  constexpr std::size_t size() const noexcept {
    return Size;
  }

  /// Construct an array
  partition_ndarray() {
    // Add the intrinsic according expressing to the target compiler the
    // partitioning to use
    if constexpr (partition_type == partition::type::cyclic)
      partition::xilinx_partition_array(
          (ValueType(*)[Size])(&elems),
          partition_type, PartitionType::split_into,
          PartitionType::partition_dim);
    if constexpr (partition_type == partition::type::block)
      partition::xilinx_partition_array(
          (ValueType(*)[Size])(&elems),
          partition_type, PartitionType::split_into,
          PartitionType::partition_dim);
    if constexpr (partition_type == partition::type::complete)
      partition::xilinx_partition_array(
          (ValueType(*)[Size])(&elems),
          partition_type, 0, PartitionType::partition_dim);
  }

  /// Determine if another type has the same underlying type and dimension and
  /// size as the current type.
  template <typename OtherTy>
  static constexpr bool is_layout_compatible =
      std::is_same_v<elem_type,
                     typename std::remove_reference_t<OtherTy>::elem_type>
          &&std::is_same_v<dim_type,
                           typename std::remove_reference_t<OtherTy>::dim_type>;

  /// Construct from an N-dimension std::initializer_list
  partition_ndarray(typename recursive_type::init i) : partition_ndarray() {
    recursive_type::recursively_on_each(elems, i,
                            [](auto &self, auto other) { self = other; });
  }

  /// Construct from an other partition_ndarray with the same type and
  /// dimensions
  template <typename OtherTy> requires is_layout_compatible<OtherTy>
  partition_ndarray(OtherTy &&other) : partition_ndarray() {
    recursive_type::recursively_on_each(elems, std::forward<OtherTy>(other),
                            [](auto &self, auto other) { self = other; });
  }

  template <typename OtherTy> requires is_layout_compatible<OtherTy>
  partition_ndarray &operator=(const OtherTy &other) {
    recursive_type::recursively_on_each(elems, other,
                            [](auto &self, auto other) { self = other; });
    return *this;
  }

  template <typename OtherTy> requires is_layout_compatible<OtherTy>
  bool operator!=(const OtherTy &other) {
    return recursive_type::recursively_on_each(
        elems, other, [](auto &self, auto other) { return self != other; });
  }

  template <typename OtherTy> requires is_layout_compatible<OtherTy>
  bool operator==(const OtherTy &other) {
    return !(*this != other);
  }

  /// Provide a subscript operator
  decltype(auto) operator[](std::size_t i) {
    return elems[i];
  }

  constexpr decltype(auto) operator[](std::size_t i) const {
    return elems[i];
  }

  /// Return the partition type of the array
  constexpr auto get_partition_type() const -> decltype(partition_type) {
    return partition_type;
  }
};

}  // namespace ext::xilinx
}  // namespace sycl

}

#endif// SYCL_XILINX_FPGA_PARTITION_ARRAY_HPP
