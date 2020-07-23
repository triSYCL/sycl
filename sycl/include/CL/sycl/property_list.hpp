//==--------- property_list.hpp --- SYCL property list ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <tuple>
#include <type_traits>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// HOW TO ADD NEW PROPERTY INSTRUCTION:
// 1. Add forward declaration of property class.
// 2. Add new record in PropKind enum.
// 3. Use RegisterProp macro passing new record from enum and new class.
// 4. Add implementation of the new property class using detail::Prop class with
//    template parameter = new record in enum as a base class.

namespace property {

namespace image {
class use_host_ptr;
class use_mutex;
class context_bound;
} // namespace image

namespace buffer {
class use_host_ptr;
class use_mutex;
class context_bound;
} // namespace buffer

namespace queue {
class enable_profiling;
class in_order;
} // namespace queue

class noinit;

namespace detail {

// Will be aliased in the sycl::ext::oneapi::property namespace
namespace buffer_ {
class use_pinned_host_memory;
}

// List of all properties' IDs.
enum PropKind {
  // Buffer properties
  BufferUseHostPtr = 0,
  BufferContextBound,
  BufferUseMutex,

  // Image properties
  ImageUseHostPtr,
  ImageContextBound,
  ImageUseMutex,

  // Queue properties
  QueueEnableProfiling,
  InOrder,

  // Accessor
  NoInit,

  BufferUsePinnedHostMemory,

  PropKindSize
};

// Base class for all properties. Needed to check that user passed only
// SYCL's properties to property_list c'tor.
class PropBase {};

// Second base class, needed for mapping PropKind to class and vice versa.
template <PropKind PropKindT> class Prop;

// This class is used in property_list to hold properties.
template <class T> class PropertyHolder {
public:
  PropertyHolder() = default;

  PropertyHolder(const PropertyHolder &P) {
    if (P.isInitialized()) {
      new (m_Mem) T(P.getProp());
      m_Initialized = true;
    }
  }

  ~PropertyHolder() {
    if (m_Initialized) {
      T *MemPtr = reinterpret_cast<T *>(m_Mem);
      MemPtr->~T();
    }
  }

  PropertyHolder &operator=(const PropertyHolder &Other) {
    if (this != &Other) {
      if (m_Initialized) {
        T *MemPtr = reinterpret_cast<T *>(m_Mem);
        MemPtr->~T();
        m_Initialized = false;
      }

      if (Other.m_Initialized) {
        new (m_Mem) T(Other.getProp());
        m_Initialized = true;
      }
    }
    return *this;
  }

  void setProp(const T &Rhs) {
    new (m_Mem) T(Rhs);
    m_Initialized = true;
  }

  const T &getProp() const {
    assert(true == m_Initialized && "Property was not set!");
    const T *MemPtr = reinterpret_cast<const T *>(m_Mem);
    return *MemPtr;
  }
  bool isInitialized() const { return m_Initialized; }

private:
  // Memory that is used for property allocation
  alignas(T) unsigned char m_Mem[sizeof(T)];
  // Indicate whether property initialized or not.
  bool m_Initialized = false;
};

// This macro adds specialization of class Prop which provides possibility to
// convert PropKind to class and vice versa.
#define RegisterProp(PropKindT, Type)                                          \
  template <> class Prop<PropKindT> : public PropBase {                        \
  public:                                                                      \
    static constexpr PropKind getKind() { return PropKindT; }                  \
    using FinalType = Type;                                                    \
  }

// Image
RegisterProp(PropKind::ImageUseHostPtr, image::use_host_ptr);
RegisterProp(PropKind::ImageUseMutex, image::use_mutex);
RegisterProp(PropKind::ImageContextBound, image::context_bound);

// Buffer
RegisterProp(PropKind::BufferUseHostPtr, buffer::use_host_ptr);
RegisterProp(PropKind::BufferUseMutex, buffer::use_mutex);
RegisterProp(PropKind::BufferContextBound, buffer::context_bound);
RegisterProp(PropKind::BufferUsePinnedHostMemory,
             buffer_::use_pinned_host_memory);

// Queue
RegisterProp(PropKind::QueueEnableProfiling, queue::enable_profiling);
RegisterProp(PropKind::InOrder, queue::in_order);

// Accessor
RegisterProp(PropKind::NoInit, noinit);

// Sentinel, needed for automatic build of tuple in property_list.
RegisterProp(PropKind::PropKindSize, PropBase);

// Common class for use_mutex in buffer and image namespaces.
template <PropKind PropKindT> class UseMutexBase : public Prop<PropKindT> {
public:
  UseMutexBase(mutex_class &MutexRef) : m_MutexClass(MutexRef) {}
  mutex_class *get_mutex_ptr() const { return &m_MutexClass; }

private:
  mutex_class &m_MutexClass;
};

// Common class for context_bound in buffer and image namespaces.
template <PropKind PropKindT> class ContextBoundBase : public Prop<PropKindT> {
public:
  ContextBoundBase(cl::sycl::context Context) : m_Context(Context) {}
  context get_context() const { return m_Context; }

private:
  cl::sycl::context m_Context;
};
} // namespace detail

namespace image {

class use_host_ptr : public detail::Prop<detail::PropKind::ImageUseHostPtr> {};

class use_mutex : public detail::UseMutexBase<detail::PropKind::ImageUseMutex> {
public:
  use_mutex(mutex_class &MutexRef) : UseMutexBase(MutexRef) {}
};

class context_bound
    : public detail::ContextBoundBase<detail::PropKind::ImageContextBound> {
public:
  context_bound(cl::sycl::context Context) : ContextBoundBase(Context) {}
};

} // namespace image

namespace buffer {

class use_host_ptr : public detail::Prop<detail::PropKind::BufferUseHostPtr> {};

class use_mutex
    : public detail::UseMutexBase<detail::PropKind::BufferUseMutex> {
public:
  use_mutex(mutex_class &MutexRef) : UseMutexBase(MutexRef) {}
};

class context_bound
    : public detail::ContextBoundBase<detail::PropKind::BufferContextBound> {
public:
  context_bound(cl::sycl::context Context) : ContextBoundBase(Context) {}
};

} // namespace buffer

namespace detail {
namespace buffer_ {
class use_pinned_host_memory
    : public detail::Prop<detail::PropKind::BufferUsePinnedHostMemory> {};
} // namespace buffer_
} // namespace detail

namespace queue {
class enable_profiling
    : public detail::Prop<detail::PropKind::QueueEnableProfiling> {};

class in_order : public detail::Prop<detail::PropKind::InOrder> {};
} // namespace queue

class noinit : public detail::Prop<detail::PropKind::NoInit> {};

} // namespace property

namespace ext {
namespace oneapi {
namespace property {
namespace buffer {
using use_pinned_host_memory =
    sycl::property::detail::buffer_::use_pinned_host_memory;
} // namespace buffer
} // namespace property
} // namespace oneapi
} // namespace ext

#if __cplusplus > 201402L

inline constexpr property::noinit noinit;

#else

namespace {

constexpr const auto &noinit =
    sycl::detail::InlineVariableHelper<property::noinit>::value;

}

#endif

class property_list {

  // The structs validate that all objects passed are base of PropBase class.
  template <typename... Tail> struct AllProperties : std::true_type {};
  template <typename T, typename... Tail>
  struct AllProperties<T, Tail...>
      : std::conditional<std::is_base_of<property::detail::PropBase, T>::value,
                         AllProperties<Tail...>, std::false_type>::type {};

  template <class T>
  using PropertyHolder = cl::sycl::property::detail::PropertyHolder<T>;
  template <property::detail::PropKind PropKindT>
  using Property = cl::sycl::property::detail::Prop<PropKindT>;

  // The structs build tuple type that can hold all properties.
  template <typename... Head> struct DefineTupleType {
    using Type = std::tuple<Head...>;
  };

  template <int Counter, typename... Head>
  struct BuildTupleType
      : public std::conditional<
            (Counter < property::detail::PropKind::PropKindSize),
            BuildTupleType<
                Counter + 1, Head...,
                PropertyHolder<typename Property<(property::detail::PropKind)(
                    Counter)>::FinalType>>,
            DefineTupleType<Head...>>::type {};

public:
  // C'tor initialize m_PropList with properties passed by invoking ctorHelper
  // recursively
  template <typename... propertyTN,
            typename = typename std::enable_if<
                AllProperties<propertyTN...>::value>::type>
  property_list(propertyTN... Props) {
    ctorHelper(Props...);
  }

  template <typename propertyT> propertyT get_property() const {
    if (!has_property<propertyT>()) {
      throw sycl::invalid_object_error();
    }
    const auto &PropHolder =
        std::get<static_cast<int>(propertyT::getKind())>(m_PropsList);
    return PropHolder.getProp();
  }

  template <typename propertyT> bool has_property() const {
    if (static_cast<int>(propertyT::getKind()) >
        property::detail::PropKind::PropKindSize)
      return false;
    return std::get<(int)(propertyT::getKind())>(m_PropsList).isInitialized();
  }

private:
  void ctorHelper() {}

  template <typename... propertyTN, class PropT>
  void ctorHelper(PropT &Prop, propertyTN... props) {
    std::get<static_cast<int>(PropT::getKind())>(m_PropsList).setProp(Prop);
    ctorHelper(props...);
  }

  // Tuple that able to hold all the properties.
  BuildTupleType<0>::Type m_PropsList;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
