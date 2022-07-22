//==---------- pi_xrt.cpp - XRT Plugin -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_xrt.cpp
/// Implementation of XRT Plugin.
///
/// \ingroup sycl_pi_xrt

#ifndef __linux__
#error "unsupported OS"
#endif

#if !defined(__clang__) && (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

/// XRT has a bunch of very noisy warnings we need to suppress.
/// XRT can be included also as an OpenCL header implementation, so we suppress
/// warning on both XRT native header and OpenCL headers
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wc++98-compat-extra-semi"
#pragma clang diagnostic ignored "-Wgnu-include-next"
#endif

#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/terminate_xsimk.hpp>
#include <CL/cl.h>

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdint.h>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <experimental/xrt_system.h>
#include <experimental/xrt_xclbin.h>
#include <version.h>
#include <xclbin.h>
#include <xrt.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>
#include <xrt_mem.h>

#include "reproducer.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

// Most of the functions in the API exist but do not have an implementation, so most
// parameters are not used and it is expected
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#elif defined(__GNUC__)

#endif

/// Base class for _pi_* objects that need ref-counting 
template <typename base> class ref_counted_base {
  std::atomic<uint32_t> refCount_;

public:
  // ref_counted_base should always be built via make_ref_counted
  // the pointer will be wrapped into a ref_counted_ref that will increment the
  // ref_count
  ref_counted_base() : refCount_(0) {}
  ref_counted_base(const ref_counted_base &) = delete;
  ref_counted_base &operator=(const ref_counted_base &) = delete;

  uint32_t get_reference_count() const noexcept { return refCount_; }

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  /// if the result is 0 the object has been destroyed
  uint32_t decrement_reference_count() noexcept {
    assert(refCount_ > 0);
    uint32_t ref_count = --refCount_;
    if (ref_count == 0) {
      delete static_cast<base *>(this);
    }
    return ref_count;
  }
  ~ref_counted_base() { assert(refCount_ == 0); }
};

/// Will increment ref count if needed by the object
template <typename T>
std::void_t<decltype(std::declval<T>().increment_reference_count())>
incr_ref_count(T *value) {
  assert(value);
  value->increment_reference_count();
}
void incr_ref_count(void *) {}

/// Will decrement ref count if needed by the object
template <typename T>
std::void_t<decltype(std::declval<T>().decrement_reference_count())>
decr_ref_count(T *value) {
  assert(value);
  value->decrement_reference_count();
}
void decr_ref_count(void *) {}

/// Validate a _pi_* object.
template <typename T, typename std::enable_if_t<
                          std::is_base_of_v<ref_counted_base<T>, T>, int> = 0>
void assert_valid_obj(T *obj) {
  assert(obj);
  assert(obj->get_reference_count() > 0);
}

template <typename T, typename std::enable_if_t<
                          !std::is_base_of_v<ref_counted_base<T>, T>, int> = 0>
void assert_valid_obj(T *obj) {
  assert(obj);
}

template <typename T> void assert_valid_objs(T ptr, int count) {
  for (int i = 0; i < count; i++)
    assert_valid_obj(ptr[i]);
}

/// ref-counting reference to a _pi_* object.
/// This should to used to reference all _pi_* objects, it will fallback if the
/// object has no ref-counting.
template <typename T> class ref_counted_ref {
  T *value = nullptr;

public:
  ref_counted_ref() = default;
  ref_counted_ref(std::nullptr_t) : ref_counted_ref() {}

  /// Copy from an existing ref so we increase the ref count
  ref_counted_ref(T *v) : value(v) { retain(); }
  ref_counted_ref(const ref_counted_ref &other)
      : ref_counted_ref(other.value) {}
  ref_counted_ref(ref_counted_ref &&other) : value(other.value) {
    other.value = nullptr;
  }

  T *operator->() { return value; }
  T &operator*() { return *value; }
  operator bool() { return value; }
  T *get() { return value; }

  /// to be used when providing a raw pointer to the user of the PI
  T *give_externally() {
    retain();
    return value;
  }

  void release() {
    if (value)
      decr_ref_count(value);
  }
  void retain() {
    if (value)
      incr_ref_count(value);
  }
  void reset() {
    release();
    value = nullptr;
  }

  void swap(ref_counted_ref &other) { std::swap(value, other.value); }

  friend bool operator==(const ref_counted_ref self, const T *other) {
    return self.value == other;
  }
  friend bool operator==(const ref_counted_ref self,
                         const ref_counted_ref other) {
    return self == other.value;
  }
  friend bool operator!=(const ref_counted_ref self, const T *other) {
    return !(self == other);
  }
  friend bool operator!=(const ref_counted_ref self,
                         const ref_counted_ref other) {
    return !(self == other);
  }

  ref_counted_ref &operator=(ref_counted_ref other) {
    swap(other);
    return *this;
  }
  ~ref_counted_ref() { release(); }
};

/// Make a new _pi_* object
template <typename base, typename... Args>
ref_counted_ref<base> make_ref_counted(Args &&...args) {
  return new base(std::forward<Args>(args)...);
}

/// Used for _pi_* objects that have at most one instance at all times
template <typename T> struct unique {
private:
  /// Storage for the instance
  inline static T *self = nullptr;

public:
  unique() = default;
  unique(const unique &) = delete;
  unique &operator=(const unique &) = delete;

  /// Return a new instance or the currently live instance
  template <typename... Ts> static ref_counted_ref<T> get(Ts &&...ts) {
    if (!self) {
      auto res = make_ref_counted<T>(std::forward<Ts>(ts)...);
      self = res.get();
      return res;
    }
    return self;
  }
  ~unique() {
    /// Make null such that the next request will rebuild
    self = nullptr;
  }
};

/// Intrusive list mix-in
template <typename T> struct intr_list_node {
  intr_list_node<T> *prev = nullptr;
  T *next = nullptr;

  void verify_invariance() {
    assert(!prev || prev->next == this);
    assert(!next || next->prev == this);
  }

  void insert_next(T *other) {
    verify_invariance();
    assert(other->next == nullptr);
    assert(other->prev == nullptr);
    other->next = next;
    other->prev = this;
    if (next)
      next->prev = other;
    next = other;
    next->verify_invariance();
    verify_invariance();
  }

  void unlink_self() {
    verify_invariance();
    if (!prev) {
      assert(!next);
      return;
    }
    prev->next = next;
    if (next)
      next->prev = prev;
    prev = nullptr;
    next = nullptr;
  }

  ~intr_list_node() { unlink_self(); }
};

/// Cannot be used for XRT objects because xrt also has global destructors
struct cleanup_system {
  using cleanup_item = std::pair<void (*)(void *), void *>;
  std::vector<cleanup_item> cleanup_list;
  std::unordered_map<void *, unsigned> cleanup_map;

  template <typename T> void remove(T *ptr) {
    unsigned &idx = cleanup_map[ptr];
    /// remove if in the cleanup list
    /// should we assert ?
    if (cleanup_list[idx].second == ptr)
      cleanup_list[idx] = cleanup_item{nullptr, nullptr};
  }

  template <typename T> void add(T *ptr) {
    unsigned &idx = cleanup_map[ptr];
    assert(idx == 0);
    /// Update the map
    idx = cleanup_list.size();
    cleanup_list.push_back(cleanup_item{[](void *p) { ((T *)p)->~T(); }, ptr});
  }
  ~cleanup_system() {
    /// idx must be signed
    for (int idx = cleanup_list.size() - 1; idx >= 0; idx--)
      if (cleanup_list[idx].second)
        cleanup_list[idx].first(cleanup_list[idx].second);
  }
} cleanup;

struct workqueue {
  std::deque<std::function<void()>> cmds;

  void clear_queue() { cmds.clear(); }
  void exec_queue() {
    for (auto &func : cmds)
      func();
    clear_queue();
  }
};

/// get an XRT object of type To from an opaque handle
template <typename To>
typename std::enable_if<std::is_reference_v<To>, To>::type
from_native_handle(pi_native_handle handle) {
  return REPRODUCE_ADD_EXTERNAL(
      *reinterpret_cast<std::remove_reference_t<To> *>(handle));
}

/// get an opaque handle out of am XRT object of type From
template <typename From>
typename std::enable_if<std::is_reference_v<From>, pi_native_handle>::type
to_native_handle(From &&from) {
  return reinterpret_cast<pi_native_handle>(std::addressof(from));
}

struct _pi_device
/// TODO: _pi_device should be ref-counted , but the SYCL runtime
/// seems to expect piDevicesGet to return objects with a ref-count
/// of 0 so we do not destroy _pi_device for now.
/// (https://github.com/intel/llvm/issues/6034)
/// : ref_counted_base<_pi_device>
{
  using native_type = xrt::device;

private:
  native_type xrtDevice_;
  /// _pi_platform holds a counting reference onto all its _pi_device so we do
  /// not keep counting reference on devices to prevent circular dependency.
  _pi_platform *platform_;

public:
  _pi_device(native_type dev, _pi_platform *platform)
      : xrtDevice_(std::move(dev)), platform_(platform) {}

  native_type &get_native() noexcept { return xrtDevice_; };
  ref_counted_ref<_pi_platform> get_platform() const noexcept {
    return platform_;
  };
};

/// A PI platform stores all known PI devices,
///  in the XRT plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct _pi_platform : ref_counted_base<_pi_platform>, unique<_pi_platform> {
private:
  std::vector<ref_counted_ref<_pi_device>> devices_;

public:
  _pi_platform() {
    int device_count = xrt::system::enumerate_devices();
    devices_.reserve(device_count);
    for (int idx = 0; idx < device_count; idx++)
      devices_.emplace_back(
          make_ref_counted<_pi_device>(REPRODUCE_CALL(xrt::device, idx), this));
  }
  unsigned get_num_device() { return devices_.size(); }
  ref_counted_ref<_pi_device> get_device(unsigned idx) { return devices_[idx]; }
  /// Add a device if it inst't already in the list
  template <typename... Ts> ref_counted_ref<_pi_device> make_device(Ts &&...ts) {
    auto new_dev = REPRODUCE_CALL(xrt::device, std::forward<Ts>(ts)...);
    auto bdf = new_dev.template get_info<xrt::info::device::bdf>();
    for (ref_counted_ref<_pi_device> dev : devices_)
      if (bdf == dev->get_native().get_info<xrt::info::device::bdf>())
        return dev;
    auto dev_ref = make_ref_counted<_pi_device>(std::move(new_dev), this);
    devices_.emplace_back(dev_ref.get());
    return dev_ref;
  }
};

struct _pi_context : ref_counted_base<_pi_context>, unique<_pi_context> {
  std::vector<ref_counted_ref<_pi_device>> devices_;

  _pi_context(uint32_t num_devices, const pi_device *devices)
      : devices_(devices, devices + num_devices) {}
};

struct _pi_mem : ref_counted_base<_pi_mem> {
  using native_type = xrt::bo;

  // Context where the memory object is accessible
  ref_counted_ref<_pi_context> context_;
  struct _mem {
    pi_mem_flags flags;
    size_t size;
    void *host_ptr;
    void *mapped_ptr;
    /// only used for asserts
    void *dev_ptr;
  } mem;

  /// Writes cannot be performed on a buffer until the xclbin has been loaded.
  /// But SYCL can request some writes before loading the xclbin so we just
  /// enqueue them to process them later.
  workqueue pending_cmds;

  _pi_mem(_pi_context *ctx, _mem m) : context_(ctx), mem(m) {}

  template <typename T>
  void run_when_mapped(const xrt::device &device, T &&call) {
    // TODO: kept as an if def because we have some an open bug in XRT that we
    // may be asked to reproduce (https://github.com/Xilinx/XRT/issues/6589)
#if 1
    if (is_mapped(device)) // TODO fix this
      std::forward<T>(call)();
    else
      pending_cmds.cmds.push_back(std::forward<T>(call));
#else
    std::forward<T>(call)();
#endif
  }

  bool is_mapped(const xrt::device &device) {
    if (mem.mapped_ptr) {
      assert(device.get_handle().get() == mem.dev_ptr &&
             "can only be mapped on one device for now");
      return true;
    }
    return false;
  }

  void map_if_needed(const xrt::device &device, xrt::memory_group grp) {
    if (mem.mapped_ptr) {
      assert(device.get_handle().get() == mem.dev_ptr &&
             "can only be mapped on one device for now");
      return;
    }
    mem.dev_ptr = device.get_handle().get();
    if (mem.host_ptr)
      get_native() =
          REPRODUCE_CALL(xrt::bo, device, mem.host_ptr, mem.size, grp);
    else
      get_native() =
          REPRODUCE_CALL(xrt::bo, device, mem.size, XRT_BO_FLAGS_NONE, grp);
    mem.mapped_ptr = REPRODUCE_MEMCALL(get_native(), map);
    pending_cmds.exec_queue(); // TODO fix this
  }

  // TODO: xrt::bo sometimes stay stuck while being deleted. So we do not delete
  // it. (https://github.com/Xilinx/XRT/issues/6588)
  // TODO: no_destroy should be a template wrapper.
  union no_destroy {
    native_type buffer_ = {};
    ~no_destroy() {}
  } nd;

  native_type &get_native() { return nd.buffer_; }

  ~_pi_mem() { assert(pending_cmds.cmds.empty()); }
};

/// The Pi calls are never queued into the _pi_queue
/// So this is almost empty
struct _pi_queue : ref_counted_base<_pi_queue> {
  ref_counted_ref<_pi_context> context_;
  ref_counted_ref<_pi_device> device_;
  intr_list_node<_pi_event> event_list;

  _pi_queue(_pi_context *context, _pi_device *device)
      : context_{context}, device_{device} {}
  /// iterator over all events in the list, it is safe to modify the provided
  /// list node in func.
  template<typename T>
  void for_each_events(T func);
};

using pfn_notify = void (*)(pi_event event, pi_int32 eventCommandStatus,
                            void *userData);

struct _pi_event : ref_counted_base<_pi_event>, intr_list_node<_pi_event> {
protected:
  _pi_event(_pi_queue *q) { q->event_list.insert_next(this); }

public:
  _pi_event() {}
  virtual void wait() {}
  virtual bool is_done() { return true; }
  virtual ~_pi_event() {}
};

template <typename T> void _pi_queue::for_each_events(T func) {
  _pi_event *curr = event_list.next;
  while (curr) {
    /// Make sure we are not affected by changes to the node made by the function
    _pi_event *next = curr->next;
    func(curr);
    curr = next;
  }
}

/// Implementation of PI Program
struct _pi_program : ref_counted_base<_pi_program> {
  ref_counted_ref<_pi_context> context_;
  xrt::xclbin bin_;

  _pi_program(pi_context ctx, xrt::xclbin bin)
      : context_(ctx), bin_(std::move(bin)) {}
  ref_counted_ref<_pi_device> get_device() {
    assert(context_->devices_.size() == 1);
    return context_->devices_[0];
  }
  xrt::uuid get_uuid() { return REPRODUCE_MEMCALL(bin_, get_uuid); }
};

/// Implementation of a PI Kernel for XRT
///
struct _pi_kernel : ref_counted_base<_pi_kernel> {
  using native_type = xrt::kernel;

  ref_counted_ref<_pi_program> prog_;
  native_type kernel_;
  xrt::run run_;
  xrt::xclbin::kernel info_;

  _pi_kernel(ref_counted_ref<_pi_program> ctx, native_type kern,
             xrt::xclbin::kernel info)
      : prog_(ctx), kernel_(std::move(kern)),
        run_(REPRODUCE_CALL(xrt::run, kernel_)), info_(std::move(info)) {}
  ref_counted_ref<_pi_context> get_context() { return prog_->context_; }
  ref_counted_ref<_pi_device> get_device() {
    assert(get_context()->devices_.size() == 1);
    return get_context()->devices_[0];
  }
};

void wait_on_events(const _pi_event* const* e, int count) {
  for (int i = 0; i < count; i++)
    const_cast<_pi_event*>(e[i])->wait();
}

// -------------------------------------------------------------
// Helper types and functions
//

namespace {

/// \cond NODOXY
template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value) {

    if (param_value_size < value_size) {
      return PI_ERROR_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    // Ignore unused parameter
    (void)value_size;

    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

} // anonymous namespace

/// ------ Error handling, matching OpenCL plugin semantics.
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::detail::pi {

std::ostream &log() { return std::cerr; }

// Report error and no return (keeps compiler from printing warnings).
//
[[noreturn]] void die(const char *Message) {
  log() << "pi_die: " << Message << std::endl;
  std::terminate();
}

void assertion(bool Condition, const char *Message) {
  if (!Condition) {
    die(Message);
  }
}

/// sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
[[noreturn]] void unimplemented(const std::string str) {

  log() << "pi_xrt: unimplemented: call to unimplemented function in XRT "
           "backend: "
        << str << std::endl;

  std::terminate();
}

}  // namespace sycl::detail::pi
} // __SYCL_INLINE_NAMESPACE(cl)

/// Check that we always use the same thread to call this function
void assert_single_thread() {
  static std::thread::id saved_id = std::this_thread::get_id();
  assert(saved_id == std::this_thread::get_id() &&
         "there is no multithread support for now");
}

/// Obtains the XRT platform.
pi_result xrt_piPlatformsGet(uint32_t num_entries, pi_platform *platforms,
                             uint32_t *num_platforms) {
  pi_result result = PI_SUCCESS;

  if (num_entries == 0 && platforms)
    return PI_ERROR_INVALID_VALUE;
  if (platforms == nullptr && num_platforms == nullptr)
    return PI_ERROR_INVALID_VALUE;
  if (num_platforms)
    *num_platforms = 1; // only 1 platform
  if (platforms)
    *platforms = _pi_platform::get().give_externally();
  return result;
}

pi_result xrt_piPlatformGetInfo(pi_platform platform,
                                pi_platform_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  assert_valid_obj(platform);

  switch (param_name) {
  case PI_PLATFORM_INFO_NAME:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "XILINX XRT BACKEND");
  case PI_PLATFORM_INFO_VENDOR:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "Xilinx Corporation");
  case PI_PLATFORM_INFO_PROFILE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "FULL PROFILE");
  case PI_PLATFORM_INFO_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   xrt_build_version);
  }
  case PI_PLATFORM_INFO_EXTENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }
  sycl::detail::pi::die("Platform info request not implemented");
  return {};
}

pi_result xrt_piextPlatformGetNativeHandle(pi_platform platform,
                                           pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextPlatformCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                  pi_platform *platform) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// \param devices List of devices available on the system
/// \param num_devices Number of elements in the list of devices
/// Requesting a non-accelerator device triggers an error, all PI XRT devices
/// are accelerators.
///
pi_result xrt_piDevicesGet(pi_platform platform, pi_device_type device_type,
                           uint32_t num_entries, pi_device *devices,
                           uint32_t *num_devices) {
  assert_valid_obj(platform);

  if (num_devices) {
    *num_devices = platform->get_num_device();
  }

  if (devices) {
    for (size_t i = 0; i < platform->get_num_device(); ++i)
      devices[i] = platform->get_device(i).give_externally();
  }

  return PI_SUCCESS;
}

/// \return PI_SUCCESS if the function is executed successfully
/// XRT devices are always root devices so retain always returns success.
pi_result xrt_piDeviceRetain(pi_device dev) {
  incr_ref_count(dev);
  return PI_SUCCESS;
}

/// \return PI_SUCCESS always since XRT devices are always root devices.
pi_result xrt_piDeviceRelease(pi_device dev) {
  decr_ref_count(dev);
  return PI_SUCCESS;
}

pi_result xrt_piDeviceGetInfo(pi_device device, pi_device_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  /// TODO: should be assert_valid_obj
  assert(device);
  auto native_dev = device->get_native();

  switch (param_name) {
  case PI_DEVICE_INFO_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_DEVICE_TYPE_ACC);
  }
  case PI_DEVICE_INFO_VENDOR_ID: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0);
  }
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   uint32_t{0});
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  }

  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1});
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   uint32_t{1});
  }
  case PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, false);
  }

  case PI_DEVICE_INFO_ATOMIC_64: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_MEMORY_ORDER_RELAXED);
  }
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    return getInfo(
        param_value_size, param_value, param_value_size_ret,
        static_cast<int>(
            native_dev.get_info<xrt::info::device::max_clock_frequency_mhz>()));
  }
  case PI_DEVICE_INFO_ADDRESS_BITS: {
    auto bits = uint32_t{std::numeric_limits<uintptr_t>::digits};
    return getInfo(param_value_size, param_value, param_value_size_ret, bits);
  }
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{32 << 20});
  }
  case PI_DEVICE_INFO_IMAGE_SUPPORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_FALSE);
  }
  // No support for Images
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
  case PI_DEVICE_INFO_MAX_SAMPLERS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, int{0});
  }
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // Does not make much sense
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{4 << 10});
  }
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    // TODO: adapt to device
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   4 << (10 + 3));
  }
  case PI_DEVICE_INFO_HALF_FP_CONFIG: {
    // TODO: Depends on the configuration
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG: {
    // TODO: Depends on the configuration
    auto config = PI_FP_INF_NAN | PI_FP_ROUND_TO_NEAREST;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    // TODO: Depends on the configuration
    auto config = PI_FP_INF_NAN | PI_FP_ROUND_TO_NEAREST;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    // TODO: Depends on the configuration
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_DEVICE_MEM_CACHE_TYPE_NONE);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // TODO: Depends on the configuration
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    // TODO: Depends on the configuration
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{0});
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    // TODO? parse memory json from XRT device api to get this info
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{0});
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    // TODO: get something device specific
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(4 << 20));
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: get something device specific
    return getInfo(param_value_size, param_value, param_value_size_ret, 8u);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE: {
    // Does not make sense
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(16 << 10));
  }
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    // TODO get something device specific
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_FALSE);
  }
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Hard coded to value returned by clinfo for Alveo U200
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1u});
  }
  case PI_DEVICE_INFO_ENDIAN_LITTLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_COMPILER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_FALSE);
  }
  case PI_DEVICE_INFO_LINKER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_FALSE);
  }
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    auto capability = CL_EXEC_NATIVE_KERNEL;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    auto capability =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
    // The mandated minimum capability:
    auto capability = CL_QUEUE_PROFILING_ENABLE;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_PLATFORM: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   device->get_platform().give_externally());
  }
  case PI_DEVICE_INFO_NAME: {
    auto name = native_dev.get_info<xrt::info::device::name>();
    return getInfoArray(strlen(name.c_str()) + 1, param_value_size, param_value,
                        param_value_size_ret, name.c_str());
  }
  case PI_DEVICE_INFO_VENDOR: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "Xilinx");
  }
  case PI_DEVICE_INFO_DRIVER_VERSION: {
    auto version = xrt_build_version;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   version);
  }
  case PI_DEVICE_INFO_PROFILE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "EMBEDDED_PROFILE");
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT: {
    // No subdevice support as of now
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   uint32_t{1});
  }
  case PI_DEVICE_INFO_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "PI 0.0");
  }
  case PI_DEVICE_INFO_OPENCL_C_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_EXTENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    // The minimum value for the FULL profile is 1 MB.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1024u});
  }
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_PARENT_DEVICE: {
    // No subdevice support as of now
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  }
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<cl_device_partition_property>(0u));
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_PARTITION_TYPE: {
    // TODO: uncouple from OpenCL
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<cl_device_partition_property>(0u));
  }
  case PI_DEVICE_INFO_BUILD_ON_SUBDEVICE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_FALSE);

    // Intel USM extensions

  case PI_DEVICE_INFO_USM_HOST_SUPPORT: {
    // from cl_intel_unified_shared_memory: "The host memory access capabilities
    // apply to any host allocation."
    //
    // query if/how the device can access page-locked host memory, possibly
    // through PCIe, using the same pointer as the host
    pi_bitfield value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    pi_bitfield value = PI_USM_ACCESS;
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The single device shared memory access capabilities apply to any shared
    // allocation associated with this device."
    //
    // query if/how the device can access managed memory associated to it
    pi_bitfield value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The cross-device shared memory access capabilities apply to any shared
    // allocation associated with this device, or to any shared memory
    // allocation on another device that also supports the same cross-device
    // shared memory access capability."
    //
    // query if/how the device can access managed memory associated to other
    // devices
    pi_bitfield value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The shared system memory access capabilities apply to any allocations
    // made by a system allocator, such as malloc or new."
    //
    // query if/how the device can access pageable host memory allocated by the
    // system allocator
    pi_bitfield value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }

    // TODO: Investigate if this information is available via XRT.
  case PI_DEVICE_INFO_PCI_ADDRESS:
  case PI_DEVICE_INFO_GPU_EU_COUNT:
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case PI_DEVICE_INFO_GPU_SLICES:
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
  case PI_DEVICE_INFO_UUID:
    return PI_ERROR_INVALID_VALUE;

  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }
  sycl::detail::pi::die("Device info request not implemented");
  return {};
}

pi_result xrt_piContextGetInfo(pi_context context, pi_context_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  assert_valid_obj(context);

  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  case PI_CONTEXT_INFO_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   context->devices_.data());
  case PI_CONTEXT_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   context->get_reference_count());
  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }

  return PI_ERROR_OUT_OF_RESOURCES;
}

pi_result xrt_piContextRetain(pi_context context) {
  incr_ref_count(context);
  return PI_SUCCESS;
}

pi_result xrt_piextContextSetExtendedDeleter(
    pi_context context, pi_context_extended_deleter function, void *user_data) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Not applicable to XRT, devices cannot be partitioned.
///
pi_result xrt_piDevicePartition(pi_device, const cl_device_partition_property *,
                                uint32_t, pi_device *, uint32_t *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Gets the native XRT handle of a PI device object
///
/// \param[in] device The PI device to get the native XRT object of.
/// \param[out] nativeHandle Set to the native handle of the PI device object.
///
/// \return PI_SUCCESS
pi_result xrt_piextDeviceGetNativeHandle(pi_device device,
                                         pi_native_handle *nativeHandle) {
  assert_valid_obj(device);

  *nativeHandle = to_native_handle<xrt::device &>(device->get_native());
  return PI_SUCCESS;
}

/// Created a PI device object from a XRT device handle.
/// \param[in] nativeHandle The native handle to create PI device object from.
/// \param[in] platform is the PI platform of the device.
/// \param[out] device Set to the PI device object created from native handle.
///
/// \return PI_SUCCESS
pi_result xrt_piextDeviceCreateWithNativeHandle(pi_native_handle handle,
                                                pi_platform platform,
                                                pi_device *res) {
  /// the SYCL runtime almost always provide a null platform
  assert(res);
  *res = _pi_platform::get()
             ->make_device(from_native_handle<xrt::device &>(handle))
             .give_externally();
  return PI_SUCCESS;
}

/// Create a PI XRT context.
///
/// \param[in] properties 0 terminated array of key/id-value combinations. Can
/// be nullptr.
/// \param[in] num_devices Number of devices to create the context for.
/// \param[in] devices Devices to create the context for.
/// \param[in] pfn_notify Callback, currently unused.
/// \param[in] user_data User data for callback.
/// \param[out] retcontext Set to created context on success.
///
/// \return PI_SUCCESS on success, otherwise an error return code.
pi_result xrt_piContextCreate(const pi_context_properties *properties,
                              uint32_t num_devices, const pi_device *devices,
                              void (*pfn_notify)(const char *errinfo,
                                                 const void *private_info,
                                                 size_t cb, void *user_data),
                              void *user_data, pi_context *ret) {
  assert_valid_objs(devices, num_devices);
  assert(devices);
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  assert(ret);
  assert(num_devices == 1);

  *ret = _pi_context::get(num_devices, devices).give_externally();
  return PI_SUCCESS;
}

pi_result xrt_piContextRelease(pi_context ctx) {
  decr_ref_count(ctx);
  return PI_SUCCESS;
}

pi_result xrt_piextContextGetNativeHandle(pi_context context,
                                          pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextContextCreateWithNativeHandle(pi_native_handle, uint32_t,
                                                 const pi_device *, bool,
                                                 pi_context *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextDeviceSelectBinary(pi_device device,
                                      pi_device_binary *binaries,
                                      uint32_t num_binaries,
                                      uint32_t *selected_binary) {
  assert_valid_obj(device);
  assert(num_binaries > 0);
  assert(selected_binary);
  (void)device;

  for (uint32_t i = 0; i < num_binaries; i++) {
    if (strncmp(binaries[i]->DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_XILINX_FPGA,
                strlen(__SYCL_PI_DEVICE_BINARY_TARGET_XILINX_FPGA)) == 0) {
      *selected_binary = i;
      return PI_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return PI_ERROR_INVALID_BINARY;
}

pi_result xrt_piextGetDeviceFunctionPointer(pi_device device,
                                            pi_program program,
                                            const char *func_name,
                                            pi_uint64 *func_pointer_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Creates a PI Memory object
///
pi_result xrt_piMemBufferCreate(pi_context context, pi_mem_flags flags,
                                size_t size, void *host_ptr, pi_mem *ret_mem,
                                const pi_mem_properties *properties) {
  assert_valid_obj(context);
  assert(ret_mem);
  assert(properties == nullptr);

  *ret_mem =
      make_ref_counted<_pi_mem>(
          context, _pi_mem::_mem{flags, size, host_ptr, nullptr, nullptr})
          .give_externally();
  return PI_SUCCESS;
}

pi_result xrt_piMemRelease(pi_mem mem) {
  decr_ref_count(mem);
  return PI_SUCCESS;
}

pi_result xrt_piMemBufferPartition(pi_mem parent_buffer, pi_mem_flags flags,
                                   pi_buffer_create_type buffer_create_type,
                                   void *buffer_create_info, pi_mem *memObj) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piMemGetInfo(pi_mem, cl_mem_info, size_t, void *, size_t *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextMemGetNativeHandle(pi_mem mem,
                                      pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextMemCreateWithNativeHandle(pi_native_handle, pi_mem *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Creates a `pi_queue` object on the XRT backend.
pi_result xrt_piQueueCreate(pi_context context, pi_device device,
                            pi_queue_properties properties, pi_queue *queue) {
  assert_valid_obj(context);
  assert_valid_obj(device);
  // TODO(XRT): : properties not handled

  *queue = make_ref_counted<_pi_queue>(context, device).give_externally();
  return PI_SUCCESS;
}

pi_result xrt_piQueueGetInfo(pi_queue command_queue, pi_queue_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piQueueRetain(pi_queue command_queue) {
  incr_ref_count(command_queue);
  return PI_SUCCESS;
}

pi_result xrt_piQueueRelease(pi_queue command_queue) {
  decr_ref_count(command_queue);
  return PI_SUCCESS;
}

pi_result xrt_piQueueFinish(pi_queue command_queue) {
  assert_valid_obj(command_queue);
  command_queue->for_each_events([](_pi_event *e) { e->wait(); });
  return PI_SUCCESS;
}

pi_result xrt_piQueueFlush(pi_queue command_queue) {
  assert_valid_obj(command_queue);
  command_queue->for_each_events([](_pi_event *e) { e->unlink_self(); });
  return PI_SUCCESS;
}

pi_result xrt_piextQueueGetNativeHandle(pi_queue queue,
                                        pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextQueueCreateWithNativeHandle(pi_native_handle, pi_context,
                                               pi_queue *,
                                               bool ownNativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                                      pi_bool blocking_write, size_t offset,
                                      size_t size, const void *ptr,
                                      uint32_t num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  assert_valid_obj(command_queue);
  assert_valid_obj(buffer);
  assert_valid_objs(event_wait_list, num_events_in_wait_list);
  assert(ptr && size);
  assert(event);
  wait_on_events(event_wait_list, num_events_in_wait_list);
  *event = make_ref_counted<_pi_event>().give_externally();

  buffer->run_when_mapped(command_queue->device_->get_native(), [=] {
    void *adjusted_ptr = ((char *)buffer->mem.mapped_ptr) + offset;
    REPRODUCE_ADD_BUFFER(ptr, size);
    REPRODUCE_CALL((void)std::memcpy, adjusted_ptr, ptr, size);
    REPRODUCE_MEMCALL(buffer->get_native(), sync, XCL_BO_SYNC_BO_TO_DEVICE);
  });

  return PI_SUCCESS;
}

pi_result xrt_piEnqueueMemBufferRead(pi_queue command_queue, pi_mem buffer,
                                     pi_bool blocking_read, size_t offset,
                                     size_t size, void *ptr,
                                     uint32_t num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  assert_valid_obj(command_queue);
  assert_valid_obj(buffer);
  assert_valid_objs(event_wait_list, num_events_in_wait_list);
  assert(ptr && size);
  assert(event);
  wait_on_events(event_wait_list, num_events_in_wait_list);
  *event = make_ref_counted<_pi_event>().give_externally();

  assert(buffer->is_mapped(command_queue->device_->get_native()));
  REPRODUCE_MEMCALL(buffer->get_native(), sync, XCL_BO_SYNC_BO_FROM_DEVICE);
  void *adjusted_ptr = ((char *)buffer->mem.mapped_ptr) + offset;
  REPRODUCE_ADD_BUFFER(ptr, size);
  REPRODUCE_CALL((void)std::memcpy, ptr, adjusted_ptr, size);
  return PI_SUCCESS;
}

pi_result xrt_piEventsWait(uint32_t num_events, const pi_event *event_list) {
  assert_valid_objs(event_list, num_events);
  /// For now every operation is executed synchronously this is a no op

  wait_on_events(event_list, num_events);

  return PI_SUCCESS;
}

pi_result xrt_piKernelCreate(pi_program program, const char *kernel_name,
                             pi_kernel *kernel) {
  assert_valid_obj(program);

  auto ker = REPRODUCE_CALL(xrt::kernel, program->get_device()->get_native(),
                            program->get_uuid(), kernel_name);
  auto info = program->bin_.get_kernel(kernel_name);
  *kernel =
      make_ref_counted<_pi_kernel>(program, std::move(ker), std::move(info))
          .give_externally();
  return PI_SUCCESS;
}

pi_result xrt_piKernelSetArg(pi_kernel kernel, uint32_t arg_index,
                             size_t arg_size, const void *arg_value) {
  assert_valid_obj(kernel);
  assert(arg_value);
  assert(arg_index < kernel->info_.get_num_args());

  REPRODUCE_ADD_BUFFER(arg_value, arg_size);
  REPRODUCE_MEMCALL(kernel->run_, set_arg, arg_index, arg_value, arg_size);
  return PI_SUCCESS;
}

pi_result xrt_piextKernelSetArgMemObj(pi_kernel kernel, uint32_t arg_index,
                                      const pi_mem *arg_value) {
  assert_valid_obj(kernel);
  assert(arg_value);
  assert(arg_index < kernel->info_.get_num_args());
  _pi_mem *buf = *arg_value;

  buf->map_if_needed(kernel->get_device()->get_native(),
                     kernel->kernel_.group_id(arg_index));
  REPRODUCE_MEMCALL(kernel->run_, set_arg, arg_index, buf->get_native());

  return PI_SUCCESS;
}

pi_result xrt_piextKernelSetArgSampler(pi_kernel kernel, uint32_t arg_index,
                                       const pi_sampler *arg_value) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                   pi_kernel_group_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueKernelLaunch(
    pi_queue command_queue, pi_kernel kernel, uint32_t work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, uint32_t num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  assert_valid_obj(command_queue);
  assert_valid_obj(kernel);
  assert(work_dim == 1 && *global_work_offset == 0 && *global_work_size == 1 &&
         *local_work_size == 1 && "only support 1 single_task");
  assert(event);
  struct _pi_event_kernel_launch : _pi_event {
    ref_counted_ref<_pi_kernel> kernel;
    bool done_flag = false;
    _pi_event_kernel_launch(_pi_queue* q, ref_counted_ref<_pi_kernel> k) : _pi_event(q), kernel(k) {}
    virtual void wait() override {
      if (done_flag)
        return;
      REPRODUCE_MEMCALL(kernel->run_, wait);
      done_flag = true;
    }
    virtual bool is_done() override {
      return done_flag;
    }
  };
  *event = make_ref_counted<_pi_event_kernel_launch>(command_queue, kernel).give_externally();

  REPRODUCE_MEMCALL(kernel->run_, start);

  return PI_SUCCESS;
}

/// \TODO Not implemented
pi_result xrt_piEnqueueNativeKernel(pi_queue, void (*)(void *), void *, size_t,
                                    uint32_t, const pi_mem *, const void **,
                                    uint32_t, const pi_event *, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextKernelGetNativeHandle(pi_kernel kernel,
                                         pi_native_handle *handle) {
  assert_valid_obj(kernel);
  assert(handle);

  *handle = to_native_handle(kernel->kernel_);

  return PI_SUCCESS;
}

pi_result xrt_piextKernelCreateWithNativeHandle(pi_native_handle handle,
                                                pi_context ctx, pi_program,
                                                bool KeepOwnership,
                                                pi_kernel *res) {
  assert_valid_obj(ctx);
  assert(ctx->devices_.size() == 1);

  xrt::kernel kern;
  if (KeepOwnership)
    kern = std::move(from_native_handle<xrt::kernel &>(handle));
  else
    kern = from_native_handle<xrt::kernel &>(handle);

  auto prog = make_ref_counted<_pi_program>(ctx, kern.get_xclbin());

  /// xrt::kernel::get_name and xrt::kernel::get_xclbin was added to xrt because
  /// we need it here. if it doesn't exist check the xrt version
  auto info = kern.get_xclbin().get_kernel(kern.get_name());
  *res = make_ref_counted<_pi_kernel>(prog, std::move(kern), std::move(info))
             .give_externally();
  return PI_SUCCESS;
}

/// \TODO Not implemented
pi_result xrt_piMemImageCreate(pi_context context, pi_mem_flags flags,
                               const pi_image_format *image_format,
                               const pi_image_desc *image_desc, void *host_ptr,
                               pi_mem *ret_mem) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// \TODO Not implemented
pi_result xrt_piMemImageGetInfo(pi_mem, pi_image_info, size_t, void *,
                                size_t *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piMemRetain(pi_mem mem) {
  incr_ref_count(mem);
  return PI_SUCCESS;
}

pi_result xrt_piclProgramCreateWithSource(pi_context, uint32_t, const char **,
                                          const size_t *, pi_program *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piProgramBuild(pi_program program, uint32_t num_devices,
                             const pi_device *device_list, const char *options,
                             void (*pfn_notify)(pi_program program,
                                                void *user_data),
                             void *user_data) {
  assert_valid_obj(program);
  assert(num_devices == 1);
  assert(device_list);
  assert_valid_obj(*device_list);
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);

  /// Noop: everything is precompiled

  return PI_SUCCESS;
}

pi_result xrt_piProgramCreate(pi_context, const void *, size_t, pi_program *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Loads the first xclbin from the list
/// Note: Only supports one device
///
pi_result xrt_piProgramCreateWithBinary(
    pi_context context, uint32_t num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *program) {
  assert_valid_obj(context);
  assert(binaries);
  assert_valid_obj(program);
  assert(num_devices == 1 && "XRT contexts are for a single device");
  assert(device_list);
  assert_valid_obj(*device_list);
  assert((context->devices_[0].get() == device_list[0]) &&
         "Mismatch between devices context and passed context when creating "
         "program from binary");

  try {
    /// We assume there is at least 1 valid device
    pi_device dev = device_list[0];
    reproducer() << "// xclbin buffer size=" << lengths[0] << "\n";
    auto xclbin = REPRODUCE_CALL(xrt::xclbin,
                                 reinterpret_cast<const axlf *>(binaries[0]));
    REPRODUCE_MEMCALL(dev->get_native(), load_xclbin, xclbin);

    *program = make_ref_counted<_pi_program>(context, std::move(xclbin))
                   .give_externally();
    return PI_SUCCESS;
  } catch (const std::system_error& err) {
    sycl::detail::pi::log()
        << "XRT error:" << err.code() << ":" << err.what() << std::endl;
    return PI_ERROR_UNKNOWN;
  }
}

pi_result xrt_piProgramGetInfo(pi_program program, pi_program_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  assert_valid_obj(program);

  switch (param_name) {
  case PI_PROGRAM_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<unsigned>(program->context_->devices_.size()));
  case PI_PROGRAM_INFO_DEVICES:
    return getInfoArray(program->context_->devices_.size(), param_value_size,
                        param_value, param_value_size_ret,
                        program->context_->devices_.data());
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->get_reference_count());
  case PI_PROGRAM_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->context_.get());
  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }
  sycl::detail::pi::die("Device info request not implemented");
  return {};
}

pi_result
xrt_piProgramLink(pi_context context, uint32_t num_devices,
                  const pi_device *device_list, const char *options,
                  uint32_t num_input_programs, const pi_program *input_programs,
                  void (*pfn_notify)(pi_program program, void *user_data),
                  void *user_data, pi_program *ret_program) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piProgramCompile(
    pi_program program, uint32_t num_devices, const pi_device *device_list,
    const char *options, uint32_t num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piProgramGetBuildInfo(pi_program program, pi_device device,
                                    cl_program_build_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {
  assert_valid_obj(program);
  assert_valid_obj(device);

  switch (param_name) {
  case PI_PROGRAM_BUILD_INFO_BINARY_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_PROGRAM_BINARY_TYPE_EXECUTABLE);
  }
  case PI_PROGRAM_BUILD_INFO_OPTIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }
}

pi_result xrt_piProgramRetain(pi_program program) {
  incr_ref_count(program);
  return PI_SUCCESS;
}

pi_result xrt_piProgramRelease(pi_program program) {
  decr_ref_count(program);
  return PI_SUCCESS;
}

pi_result xrt_piextProgramSetSpecializationConstant(pi_program prog,
                                                    uint32_t spec_id,
                                                    size_t spec_size,
                                                    const void *spec_value) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Gets the native XRT handle of a PI program object
///
/// \param[in] program The PI program to get the native XRT object of.
/// \param[out] nativeHandle Set to the native handle of the PI program object.
///
/// \return PI_SUCCESS
pi_result xrt_piextProgramGetNativeHandle(pi_program program,
                                          pi_native_handle *res) {
  assert_valid_obj(program);
  *res = to_native_handle<xrt::xclbin &>(program->bin_);
  return PI_SUCCESS;
}

/// Created a PI program object from a XRT program handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI program object from.
/// \param[in] context The PI context of the program.
/// \param[out] program Set to the PI program object created from native handle.
///
/// \return PI_SUCCESS
pi_result xrt_piextProgramCreateWithNativeHandle(pi_native_handle handle,
                                                 pi_context ctx,
                                                 bool keep_ownership,
                                                 pi_program *res) {
  assert_valid_obj(ctx);
  xrt::xclbin bin;
  if (keep_ownership)
    bin = std::move(from_native_handle<xrt::xclbin &>(handle));
  else
    bin = from_native_handle<xrt::xclbin &>(handle);
  *res = make_ref_counted<_pi_program>(ctx, std::move(bin)).give_externally();
  return PI_SUCCESS;
}

pi_result xrt_piKernelGetInfo(pi_kernel kernel, pi_kernel_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert_valid_obj(kernel);

  switch (param_name) {
  case PI_KERNEL_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   kernel->get_reference_count());
  case PI_KERNEL_INFO_CONTEXT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   kernel->get_context().give_externally());
  }
  case PI_KERNEL_INFO_PROGRAM: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   kernel->prog_.give_externally());
  }
  case PI_KERNEL_INFO_NUM_ARGS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   kernel->info_.get_num_args());
  }
  case PI_KERNEL_INFO_FUNCTION_NAME: {
    auto name = kernel->info_.get_name();
    return getInfoArray(strlen(name.c_str()) + 1, param_value_size, param_value,
                        param_value_size_ret, name.c_str());
  }
  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }

  return PI_ERROR_INVALID_VALUE;
}

pi_result xrt_piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device, pi_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piKernelRetain(pi_kernel kernel) {
  incr_ref_count(kernel);
  return PI_SUCCESS;
}

pi_result xrt_piKernelRelease(pi_kernel kernel) {
  decr_ref_count(kernel);
  return PI_SUCCESS;
}

// A NOP for the XRT backend
pi_result xrt_piKernelSetExecInfo(pi_kernel, pi_kernel_exec_info, size_t,
                                  const void *) {
  return PI_SUCCESS;
}

pi_result xrt_piextKernelSetArgPointer(pi_kernel kernel, uint32_t arg_index,
                                       size_t arg_size, const void *arg_value) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

//
// Events
//
pi_result xrt_piEventCreate(pi_context, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEventGetInfo(pi_event event, pi_event_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) {
  assert_valid_obj(event);

  switch (param_name) {
  case PI_EVENT_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   event->get_reference_count());
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_EVENT_COMPLETE);
  }
  default:
    sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
  }

  return PI_ERROR_INVALID_EVENT;
}

pi_result xrt_piEventGetProfilingInfo(pi_event event,
                                      pi_profiling_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEventSetCallback(pi_event, pi_int32, pfn_notify, void *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEventSetStatus(pi_event, pi_int32) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEventRetain(pi_event event) {
  incr_ref_count(event);
  return PI_SUCCESS;
}

pi_result xrt_piEventRelease(pi_event event) {
  decr_ref_count(event);
  return PI_SUCCESS;
}

pi_result xrt_piEnqueueEventsWait(pi_queue command_queue,
                                  uint32_t num_events_in_wait_list,
                                  const pi_event *event_wait_list,
                                  pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueEventsWaitWithBarrier(pi_queue command_queue,
                                             uint32_t num_events_in_wait_list,
                                             const pi_event *event_wait_list,
                                             pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextEventGetNativeHandle(pi_event event,
                                        pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextEventCreateWithNativeHandle(pi_native_handle, pi_context,
                                               bool, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piSamplerCreate(pi_context context,
                              const pi_sampler_properties *sampler_properties,
                              pi_sampler *result_sampler) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piSamplerGetInfo(pi_sampler sampler, cl_sampler_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piSamplerRetain(pi_sampler sampler) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piSamplerRelease(pi_sampler sampler) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Make a rectangular copy from or to the device
pi_result enqueue_rect_copy(bool is_read, pi_queue command_queue, pi_mem buffer,
                            pi_bool blocking, pi_buff_rect_offset buffer_offset,
                            pi_buff_rect_offset host_offset,
                            pi_buff_rect_region region, size_t buffer_row_pitch,
                            size_t buffer_slice_pitch, size_t host_row_pitch,
                            size_t host_slice_pitch, void *ptr,
                            uint32_t num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  assert_valid_obj(command_queue);
  assert_valid_obj(buffer);
  assert_valid_objs(event_wait_list, num_events_in_wait_list);

  wait_on_events(event_wait_list, num_events_in_wait_list);

  /// TODO add test where the offsets and sizes are not simple
  if (is_read)
    assert(buffer->is_mapped(command_queue->device_->get_native()));
  buffer->run_when_mapped(
      command_queue->device_->get_native(),
      [=, dev_off = *buffer_offset, host_off = *host_offset, size = *region] {
        if (is_read)
          REPRODUCE_MEMCALL(buffer->get_native(), sync,
                            XCL_BO_SYNC_BO_FROM_DEVICE);

        size_t dev_origin = dev_off.z_scalar * buffer_slice_pitch +
                            dev_off.y_scalar * buffer_row_pitch +
                            dev_off.x_bytes;
        size_t host_origin = host_off.z_scalar * host_slice_pitch +
                             host_off.y_scalar * host_row_pitch +
                             host_off.x_bytes;
        size_t max_host_offset =
            (host_off.z_scalar + size.depth_scalar) * host_slice_pitch +
            (host_off.y_scalar + size.height_scalar) * host_row_pitch +
            host_off.x_bytes;

        REPRODUCE_ADD_BUFFER(ptr, max_host_offset);
        for (size_t zit = 0; zit < size.depth_scalar; zit++) {
          for (size_t yit = 0; yit < size.height_scalar; yit++) {
            size_t dev_start =
                dev_origin + zit * buffer_slice_pitch + yit * buffer_row_pitch;
            size_t host_start =
                host_origin + zit * host_slice_pitch + yit * host_row_pitch;

            uint8_t *host_ptr = &((uint8_t *)(ptr))[host_start];
            uint8_t *dev_ptr =
                &((uint8_t *)(buffer->mem.mapped_ptr))[dev_start];
            REPRODUCE_ADD_RELATED_PTR(ptr, host_ptr);
            REPRODUCE_ADD_RELATED_PTR(buffer->mem.mapped_ptr, dev_ptr);
            if (is_read)
              REPRODUCE_CALL((void)std::memcpy, host_ptr, dev_ptr,
                             size.width_bytes);
            else
              REPRODUCE_CALL((void)std::memcpy, dev_ptr, host_ptr,
                             size.width_bytes);
          }
        }
        if (!is_read)
          REPRODUCE_MEMCALL(buffer->get_native(), sync,
                            XCL_BO_SYNC_BO_TO_DEVICE);
      });
  return PI_SUCCESS;
}

pi_result xrt_piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, uint32_t num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return enqueue_rect_copy(
      /*is_read*/ true, command_queue, buffer, blocking_read, buffer_offset,
      host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
      host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

pi_result xrt_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, uint32_t num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return enqueue_rect_copy(
      /*is_read*/ false, command_queue, buffer, blocking_write, buffer_offset,
      host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
      host_slice_pitch, const_cast<void *>(ptr), num_events_in_wait_list,
      event_wait_list, event);
}

pi_result xrt_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                                     pi_mem dst_buffer, size_t src_offset,
                                     size_t dst_offset, size_t size,
                                     uint32_t num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    uint32_t num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
                                     const void *pattern, size_t pattern_size,
                                     size_t offset, size_t size,
                                     uint32_t num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageRead(pi_queue command_queue, pi_mem image,
                                    pi_bool blocking_read, const size_t *origin,
                                    const size_t *region, size_t row_pitch,
                                    size_t slice_pitch, void *ptr,
                                    uint32_t num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageWrite(
    pi_queue command_queue, pi_mem image, pi_bool blocking_write,
    const size_t *origin, const size_t *region, size_t input_row_pitch,
    size_t input_slice_pitch, const void *ptr, uint32_t num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                                    pi_mem dst_image, const size_t *src_origin,
                                    const size_t *dst_origin,
                                    const size_t *region,
                                    uint32_t num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageFill(pi_queue, pi_mem, const void *,
                                    const size_t *, const size_t *, uint32_t,
                                    const pi_event *, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferMap(pi_queue command_queue, pi_mem buffer,
                                    pi_bool blocking_map,
                                    pi_map_flags map_flags, size_t offset,
                                    size_t size,
                                    uint32_t num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event, void **ret_map) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                void *mapped_ptr,
                                uint32_t num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMHostAlloc(void **result_ptr, pi_context context,
                                pi_usm_mem_properties *properties, size_t size,
                                uint32_t alignment) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                                  pi_device device,
                                  pi_usm_mem_properties *properties,
                                  size_t size, uint32_t alignment) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMSharedAlloc(void **result_ptr, pi_context context,
                                  pi_device device,
                                  pi_usm_mem_properties *properties,
                                  size_t size, uint32_t alignment) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMFree(pi_context context, void *ptr) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                    size_t count,
                                    uint32_t num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                    void *dst_ptr, const void *src_ptr,
                                    size_t size,
                                    uint32_t num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                      size_t size, pi_usm_migration_flags flags,
                                      uint32_t num_events_in_waitlist,
                                      const pi_event *events_waitlist,
                                      pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                       size_t length, pi_mem_advice advice,
                                       pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMGetMemAllocInfo(pi_context context, const void *ptr,
                                      pi_mem_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piTearDown(void *) {

  /// cleanup the potential left-overs from the hw_emu simulator
  terminate_xsimk();

  return PI_SUCCESS;
}

template <typename, auto func> struct xrt_pi_call_wrapper;

template <typename ret_ty, typename... args_ty, auto func>
struct xrt_pi_call_wrapper<ret_ty (*)(args_ty...), func> {
  static ret_ty call(args_ty... args) {
    /// Wrapper around every call to the XRT pi.
    assert_single_thread();
    try {
      return func(args...);
    } catch (...) {
      sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
      return PI_ERROR_UNKNOWN;
    }
  }
};

const char SupportedVersion[] = _PI_H_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  int CompareVersions = strcmp(PluginInit->PiVersion, SupportedVersion);
  if (CompareVersions < 0) {
    // PI interface supports lower version of PI.
    // TODO: Take appropriate actions.
    return PI_ERROR_INVALID_OPERATION;
  }

  // PI interface supports higher version or the same version.
  strncpy(PluginInit->PluginVersion, SupportedVersion, 4);

  // Set whole function table to zero to make it easier to detect if
  // functions are not set up below.
  std::memset(&(PluginInit->PiFunctionTable), 0,
              sizeof(PluginInit->PiFunctionTable));

// Forward calls to Xilinx RT (XRT).
#define _PI_CL(pi_api, xrt_api)                                                \
  (PluginInit->PiFunctionTable).pi_api = (decltype(                            \
      &::pi_api))xrt_pi_call_wrapper<decltype(&xrt_api), &xrt_api>::call;

  // Platform
  _PI_CL(piPlatformsGet, xrt_piPlatformsGet)
  _PI_CL(piPlatformGetInfo, xrt_piPlatformGetInfo)
  _PI_CL(piextPlatformGetNativeHandle, xrt_piextPlatformGetNativeHandle)
  _PI_CL(piextPlatformCreateWithNativeHandle,
         xrt_piextPlatformCreateWithNativeHandle)
  // Device
  _PI_CL(piDevicesGet, xrt_piDevicesGet)
  _PI_CL(piDeviceGetInfo, xrt_piDeviceGetInfo)
  _PI_CL(piDevicePartition, xrt_piDevicePartition)
  _PI_CL(piDeviceRetain, xrt_piDeviceRetain)
  _PI_CL(piDeviceRelease, xrt_piDeviceRelease)
  _PI_CL(piextDeviceSelectBinary, xrt_piextDeviceSelectBinary)
  _PI_CL(piextGetDeviceFunctionPointer, xrt_piextGetDeviceFunctionPointer)
  _PI_CL(piextDeviceGetNativeHandle, xrt_piextDeviceGetNativeHandle)
  _PI_CL(piextDeviceCreateWithNativeHandle,
         xrt_piextDeviceCreateWithNativeHandle)
  // Context
  _PI_CL(piContextCreate, xrt_piContextCreate)
  _PI_CL(piContextGetInfo, xrt_piContextGetInfo)
  _PI_CL(piContextRetain, xrt_piContextRetain)
  _PI_CL(piContextRelease, xrt_piContextRelease)
  _PI_CL(piextContextGetNativeHandle, xrt_piextContextGetNativeHandle)
  _PI_CL(piextContextCreateWithNativeHandle,
         xrt_piextContextCreateWithNativeHandle)
  // Queue
  _PI_CL(piQueueCreate, xrt_piQueueCreate)
  _PI_CL(piQueueGetInfo, xrt_piQueueGetInfo)
  _PI_CL(piQueueFinish, xrt_piQueueFinish)
  _PI_CL(piQueueFlush, xrt_piQueueFlush)
  _PI_CL(piQueueRetain, xrt_piQueueRetain)
  _PI_CL(piQueueRelease, xrt_piQueueRelease)
  _PI_CL(piextQueueGetNativeHandle, xrt_piextQueueGetNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle, xrt_piextQueueCreateWithNativeHandle)
  // Memory
  _PI_CL(piMemBufferCreate, xrt_piMemBufferCreate)
  _PI_CL(piMemImageCreate, xrt_piMemImageCreate)
  _PI_CL(piMemGetInfo, xrt_piMemGetInfo)
  _PI_CL(piMemImageGetInfo, xrt_piMemImageGetInfo)
  _PI_CL(piMemRetain, xrt_piMemRetain)
  _PI_CL(piMemRelease, xrt_piMemRelease)
  _PI_CL(piMemBufferPartition, xrt_piMemBufferPartition)
  _PI_CL(piextMemGetNativeHandle, xrt_piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, xrt_piextMemCreateWithNativeHandle)
  // Program
  _PI_CL(piProgramCreate, xrt_piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, xrt_piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, xrt_piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, xrt_piProgramGetInfo)
  _PI_CL(piProgramCompile, xrt_piProgramCompile)
  _PI_CL(piProgramBuild, xrt_piProgramBuild)
  _PI_CL(piProgramLink, xrt_piProgramLink)
  _PI_CL(piProgramGetBuildInfo, xrt_piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, xrt_piProgramRetain)
  _PI_CL(piProgramRelease, xrt_piProgramRelease)
  _PI_CL(piextProgramSetSpecializationConstant,
         xrt_piextProgramSetSpecializationConstant)
  _PI_CL(piextProgramGetNativeHandle, xrt_piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle,
         xrt_piextProgramCreateWithNativeHandle)
  // Kernel
  _PI_CL(piKernelCreate, xrt_piKernelCreate)
  _PI_CL(piKernelSetArg, xrt_piKernelSetArg)
  _PI_CL(piKernelGetInfo, xrt_piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, xrt_piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, xrt_piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, xrt_piKernelRetain)
  _PI_CL(piKernelRelease, xrt_piKernelRelease)
  _PI_CL(piKernelSetExecInfo, xrt_piKernelSetExecInfo)
  _PI_CL(piextKernelSetArgPointer, xrt_piextKernelSetArgPointer)
  _PI_CL(piextKernelCreateWithNativeHandle,
         xrt_piextKernelCreateWithNativeHandle)
  _PI_CL(piextKernelGetNativeHandle, xrt_piextKernelGetNativeHandle)
  // Event
  _PI_CL(piEventCreate, xrt_piEventCreate)
  _PI_CL(piEventGetInfo, xrt_piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, xrt_piEventGetProfilingInfo)
  _PI_CL(piEventsWait, xrt_piEventsWait)
  _PI_CL(piEventSetCallback, xrt_piEventSetCallback)
  _PI_CL(piEventSetStatus, xrt_piEventSetStatus)
  _PI_CL(piEventRetain, xrt_piEventRetain)
  _PI_CL(piEventRelease, xrt_piEventRelease)
  _PI_CL(piextEventGetNativeHandle, xrt_piextEventGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle, xrt_piextEventCreateWithNativeHandle)
  // Sampler
  _PI_CL(piSamplerCreate, xrt_piSamplerCreate)
  _PI_CL(piSamplerGetInfo, xrt_piSamplerGetInfo)
  _PI_CL(piSamplerRetain, xrt_piSamplerRetain)
  _PI_CL(piSamplerRelease, xrt_piSamplerRelease)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, xrt_piEnqueueKernelLaunch)
  _PI_CL(piEnqueueNativeKernel, xrt_piEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, xrt_piEnqueueEventsWait)
  _PI_CL(piEnqueueEventsWaitWithBarrier, xrt_piEnqueueEventsWaitWithBarrier)
  _PI_CL(piEnqueueMemBufferRead, xrt_piEnqueueMemBufferRead)
  _PI_CL(piEnqueueMemBufferReadRect, xrt_piEnqueueMemBufferReadRect)
  _PI_CL(piEnqueueMemBufferWrite, xrt_piEnqueueMemBufferWrite)
  _PI_CL(piEnqueueMemBufferWriteRect, xrt_piEnqueueMemBufferWriteRect)
  _PI_CL(piEnqueueMemBufferCopy, xrt_piEnqueueMemBufferCopy)
  _PI_CL(piEnqueueMemBufferCopyRect, xrt_piEnqueueMemBufferCopyRect)
  _PI_CL(piEnqueueMemBufferFill, xrt_piEnqueueMemBufferFill)
  _PI_CL(piEnqueueMemImageRead, xrt_piEnqueueMemImageRead)
  _PI_CL(piEnqueueMemImageWrite, xrt_piEnqueueMemImageWrite)
  _PI_CL(piEnqueueMemImageCopy, xrt_piEnqueueMemImageCopy)
  _PI_CL(piEnqueueMemImageFill, xrt_piEnqueueMemImageFill)
  _PI_CL(piEnqueueMemBufferMap, xrt_piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, xrt_piEnqueueMemUnmap)
  // USM
  _PI_CL(piextUSMHostAlloc, xrt_piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, xrt_piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, xrt_piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, xrt_piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, xrt_piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, xrt_piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, xrt_piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, xrt_piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMGetMemAllocInfo, xrt_piextUSMGetMemAllocInfo)

  _PI_CL(piextKernelSetArgMemObj, xrt_piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, xrt_piextKernelSetArgSampler)
  _PI_CL(piTearDown, xrt_piTearDown)

#undef _PI_CL

  return PI_SUCCESS;
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if !defined(__clang__) && (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif
