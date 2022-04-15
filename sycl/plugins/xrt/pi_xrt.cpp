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

/// XRT has a bunch of very noisy warnings we need to suppress.
/// XRT can be found as opencl implementation so we suppress warning on both xrt
/// native header and opencl headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wc++98-compat-extra-semi"
#pragma clang diagnostic ignored "-Wgnu-include-next"

#include <CL/cl.h>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/terminate_xsimk.hpp>

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

#pragma clang diagnostic pop

// Most of functions in the API exists but do not have an implementation so most
// parameters are not used and it is expected
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

/// Base class for _pi_* objects that need ref-counting
template <typename base> struct ref_counted_base {
  ref_counted_base() : refCount_(1) {}
  std::atomic_uint32_t refCount_;

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  pi_uint32 increment_reference_count() noexcept {
    assert(refCount_ > 0);
    return ++refCount_;
  }

  /// if the result is 0 the object has been destroyed
  pi_uint32 decrement_reference_count() noexcept {
    assert(refCount_ > 0);
    uint32_t ref_count = --refCount_;
    if (ref_count == 0) {
      delete static_cast<base *>(this);
    }
    return ref_count;
  }

  template <typename... Args> static base *make(Args &&...args) {
    return new base(std::forward<Args>(args)...);
  }
};

/// Will increment ref count if needed by the object
template <typename T, typename std::enable_if_t<
                          std::is_base_of_v<ref_counted_base<T>, T>, int> = 0>
void incr_ref_count(T *value) {
  if (value)
    value->increment_reference_count();
}
template <typename T, typename std::enable_if_t<
                          !std::is_base_of_v<ref_counted_base<T>, T>, int> = 0>
void incr_ref_count(T *value) {}

/// Will decrement ref count if needed by the object
template <typename T, typename std::enable_if_t<
                          std::is_base_of_v<ref_counted_base<T>, T>, int> = 0>
void decr_ref_count(T *value) {
  if (value)
    value->decrement_reference_count();
}
template <typename T, typename std::enable_if_t<
                          !std::is_base_of_v<ref_counted_base<T>, T>, int> = 0>
void decr_ref_count(T *value) {}

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

/// wrapper type for a _pi_* object that keeps a reference on an other _pi_*
/// object that might be refcounted
template <typename T> struct ref_counted_ref {
  T *value = nullptr;
  ref_counted_ref() = default;
  ref_counted_ref(std::nullptr_t) : ref_counted_ref() {}

  /// Copy from an existing ref so we increase the ref count
  ref_counted_ref(T *v) : value(v) { incr_ref_count(value); }
  ref_counted_ref(ref_counted_ref &other) : ref_counted_ref(other.value) {}
  /// Copy from an existing move the ref into the object so the ref count say
  /// the same
  ref_counted_ref(ref_counted_ref &&other) : value(other.value) {}

  T *operator->() { return value; }
  T &operator*() { return *value; }
  operator bool() { return value; }

  template <typename OtherTy> ref_counted_ref &operator=(OtherTy &&other) {
    decr_ref_count(value);
    value = ref_counted_ref(std::forward<OtherTy>(other));
    return *this;
  }
  ~ref_counted_ref() { decr_ref_count(value); }
};

struct workqueue {
  std::deque<std::function<void()>> cmds;

  void clear_queue() { cmds.clear(); }
  void exec_queue() {
    for (auto &func : cmds)
      func();
    clear_queue();
  }
};

/// A PI platform stores all known PI devices,
///  in the XRT plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct _pi_platform : ref_counted_base<_pi_platform> {
  std::vector<std::unique_ptr<_pi_device>> devices_;

  _pi_platform(unsigned int numDevices) {
    devices_.reserve(numDevices);
    for (unsigned int i = 0; i < numDevices; ++i) {
      devices_.emplace_back(
          std::make_unique<_pi_device>(REPRODUCE_CALL(xrt::device, i), *this));
    }
  }
  unsigned int getNumDevices() const noexcept { return devices_.size(); }
  _pi_device &getDevice(unsigned int i) { return *devices_[i]; }
};

struct _pi_device {
private:
  using native_type = xrt::device;

  native_type xrtDevice_;
  ref_counted_ref<_pi_platform> platform_;

public:
  _pi_device(native_type dev, _pi_platform &platform)
      : xrtDevice_(std::move(dev)), platform_(&platform) {}

  native_type &get() noexcept { return xrtDevice_; };
  pi_platform get_platform() const noexcept { return platform_.value; };
};

struct _pi_context : ref_counted_base<_pi_context> {
  struct deleter_data {
    pi_context_extended_deleter function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  std::vector<ref_counted_ref<_pi_device>> devices_;

  _pi_context(pi_uint32 num_devices, const pi_device *devices);

  ~_pi_context() = default;
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
    void *dev_ptr;
  } mem;

  /// Writes cannot be performed on a buffer until the xclbin has been loaded.
  /// But sycl can request som writes before loading the xclbin so we just
  /// enqueue them.
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
  void map_if_needed(const xrt::device &device) {
    if (mem.mapped_ptr) {
      assert(device.get_handle().get() == mem.dev_ptr &&
             "can only be mapped on one device for now");
      return;
    }
    mem.dev_ptr = device.get_handle().get();
    if (mem.host_ptr)
      get_native() = REPRODUCE_CALL(xrt::bo, device, mem.host_ptr, mem.size, 0);
    else
      get_native() =
          REPRODUCE_CALL(xrt::bo, device, mem.size, XRT_BO_FLAGS_NONE, 0);
    mem.mapped_ptr = REPRODUCE_MEMCALL(get_native(), map);
    pending_cmds.exec_queue(); // TODO fix this
  }

  // TODO: xrt::bo sometimes stay stuck while being deleted. so we dont delete
  // it. (https://github.com/Xilinx/XRT/issues/6588)
  union no_destroy {
    no_destroy() = default;
    native_type buffer_ = {};
    ~no_destroy() {}
  } nd;
  native_type &get_native() { return nd.buffer_; }
  ~_pi_mem() { assert(pending_cmds.cmds.empty()); }
};

struct _pi_queue : ref_counted_base<_pi_queue> {
  ref_counted_ref<_pi_context> context_;
  ref_counted_ref<_pi_device> device_;

  workqueue cmds;

  _pi_queue(_pi_context *context, _pi_device *device)
      : context_{context}, device_{device} {}
};

typedef void (*pfn_notify)(pi_event event, pi_int32 eventCommandStatus,
                           void *userData);

/// there are currently no async operation so event are noop
struct _pi_event : ref_counted_base<_pi_event> {};

/// Implementation of PI Program
struct _pi_program {
  ref_counted_ref<_pi_context> context_;
  ref_counted_ref<_pi_device> device_;
  xrt::xclbin bin_;
  xrt::uuid uuid_;

  _pi_program(pi_context ctx, pi_device dev, xrt::xclbin bin, xrt::uuid uuid)
      : context_(ctx), device_(dev), bin_(std::move(bin)),
        uuid_(std::move(uuid)) {}
};

/// Implementation of a PI Kernel for XRT
///
struct _pi_kernel {
  using native_type = xrt::kernel;

  ref_counted_ref<_pi_program> program_;
  native_type kernel_;
  xrt::run run_;
  xrt::xclbin::kernel info_;

  _pi_kernel(pi_program prog, native_type kern, xrt::run run,
             xrt::xclbin::kernel info)
      : program_(prog), kernel_(std::move(kern)), run_(run),
        info_(std::move(info)) {}
};

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
      return PI_INVALID_VALUE;
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
namespace sycl {
namespace detail {
namespace pi {

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
[[noreturn]] void unimplemented(const char *str) {

  log() << "pi_xrt: unimplemented: call too unimplemented function in XRT "
           "backend: "
        << str << std::endl;

  std::terminate();
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

void assert_single_thread() {
  static std::thread::id saved_id = std::this_thread::get_id();
  assert(saved_id == std::this_thread::get_id() &&
         "there is no multithread support for now");
}

/// Obtains the XRT platform.
pi_result xrt_piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                             pi_uint32 *num_platforms) {
  assert_single_thread();
  pi_result result = PI_SUCCESS;

  // run_original_test();

  // Simulators are not considered as devices by XRT...
  // And wether XRT is configured to use simulator or real device cannot be
  // changed once it is selected. So we just assume there is 1 device available.
  // And hope we are right
  static pi_uint32 numPlatforms = 1;
  static _pi_platform platform(numPlatforms);

  if (num_entries == 0 && platforms)
    return PI_INVALID_VALUE;
  if (platforms == nullptr && num_platforms == nullptr)
    return PI_INVALID_VALUE;
  if (num_platforms)
    *num_platforms = numPlatforms;
  if (platforms)
    *platforms = &platform;
  return result;
}

pi_result xrt_piPlatformGetInfo(pi_platform platform,
                                pi_platform_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  assert_single_thread();
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Platform info request not implemented");
  return {};
}

/// \param devices List of devices available on the system
/// \param num_devices Number of elements in the list of devices
/// Requesting a non-accelerator device triggers an error, all PI XRT devices
/// are accelerators.
///
pi_result xrt_piDevicesGet(pi_platform platform, pi_device_type device_type,
                           pi_uint32 num_entries, pi_device *devices,
                           pi_uint32 *num_devices) {
  assert_single_thread();
  assert_valid_obj(platform);

  pi_result err = PI_SUCCESS;
  const bool askingForDefault = device_type == PI_DEVICE_TYPE_DEFAULT;
  const bool askingForACC = device_type & PI_DEVICE_TYPE_ACC;
  const bool returnDevices = askingForDefault || askingForACC;

  size_t numDevices = returnDevices ? platform->getNumDevices() : 0;
  assert(numDevices == 1 && "XRT should only have 1 device");

  try {
    if (num_devices) {
      *num_devices = numDevices;
    }

    if (returnDevices && devices) {
      for (size_t i = 0; i < std::min(size_t(num_entries), numDevices); ++i) {
        devices[i] = &platform->getDevice(i);
      }
    }

    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

/// \return PI_SUCCESS if the function is executed successfully
/// XRT devices are always root devices so retain always returns success.
pi_result xrt_piDeviceRetain(pi_device dev) {
  assert_single_thread();
  incr_ref_count(dev);
  return PI_SUCCESS;
}

/// \return PI_SUCCESS always since XRT devices are always root devices.
pi_result xrt_piDeviceRelease(pi_device dev) {
  assert_single_thread();
  decr_ref_count(dev);
  return PI_SUCCESS;
}

pi_result xrt_piDeviceGetInfo(pi_device device, pi_device_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert_single_thread();
  assert(device);
  auto native_dev = device->get();

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
                   pi_uint32{0});
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
    auto bits = pi_uint32{std::numeric_limits<uintptr_t>::digits};
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
                   device->get_platform());
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
                   pi_uint32{1});
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
    return PI_INVALID_VALUE;

  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Device info request not implemented");
  return {};
}

pi_result xrt_piContextGetInfo(pi_context context, pi_context_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  assert_single_thread();
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_OUT_OF_RESOURCES;
}

pi_result xrt_piContextRetain(pi_context context) {
  assert_single_thread();
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
                                pi_uint32, pi_device *, pi_uint32 *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Gets the native XRT handle of a PI device object
///
/// \param[in] device The PI device to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the PI device object.
///
/// \return PI_SUCCESS
pi_result xrt_piextDeviceGetNativeHandle(pi_device device,
                                         pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Created a PI device object from a CUDA device handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI device object from.
/// \param[in] platform is the PI platform of the device.
/// \param[out] device Set to the PI device object created from native handle.
///
/// \return TBD
pi_result xrt_piextDeviceCreateWithNativeHandle(pi_native_handle, pi_platform,
                                                pi_device *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/* Context APIs */

_pi_context::_pi_context(pi_uint32 num_devices, const pi_device *devices)
    : devices_(devices, devices + num_devices) {}

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
                              pi_uint32 num_devices, const pi_device *devices,
                              void (*pfn_notify)(const char *errinfo,
                                                 const void *private_info,
                                                 size_t cb, void *user_data),
                              void *user_data, pi_context *ret) {
  assert_single_thread();
  assert_valid_objs(devices, num_devices);
  assert(devices);
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  assert(num_devices == 1);
  assert(ret);

  *ret = ref_counted_base<_pi_context>::make(num_devices, devices);
  return PI_SUCCESS;
}

pi_result xrt_piContextRelease(pi_context ctx) {
  assert_single_thread();
  decr_ref_count(ctx);
  return PI_SUCCESS;
}

/// There is no XRT context, so native handle is actually the PI context
/// \param[in] context The PI context to get the "native" XRT object of.
/// \param[out] nativeHandle Set to the "native" handle of the PI context
/// object.
///
/// \return PI_SUCCESS
pi_result xrt_piextContextGetNativeHandle(pi_context context,
                                          pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// TODO @lforg37 HERE

/// Create a PI context object from a XRT "native" context handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI context object from.
/// \param[out] context Set to the PI context object created from native handle.
///
/// \return TBD
pi_result xrt_piextContextCreateWithNativeHandle(pi_native_handle, pi_uint32,
                                                 const pi_device *, bool,
                                                 pi_context *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// \return If available, the first binary that is PTX
///
pi_result xrt_piextDeviceSelectBinary(pi_device device,
                                      pi_device_binary *binaries,
                                      pi_uint32 num_binaries,
                                      pi_uint32 *selected_binary) {
  assert_single_thread();
  assert_valid_obj(device);
  assert(num_binaries > 0);
  assert(selected_binary);
  (void)device;

  for (pi_uint32 i = 0; i < num_binaries; i++) {
    if (strncmp(binaries[i]->DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_XILINX_FPGA,
                strlen(__SYCL_PI_DEVICE_BINARY_TARGET_XILINX_FPGA)) == 0) {
      *selected_binary = i;
      return PI_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return PI_INVALID_BINARY;
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
  assert_single_thread();
  assert_valid_obj(context);
  assert(ret_mem);
  assert(properties == nullptr);

  *ret_mem = ref_counted_base<_pi_mem>::make(
      context, _pi_mem::_mem{flags, size, host_ptr, nullptr, nullptr});
  return PI_SUCCESS;
}

pi_result xrt_piMemRelease(pi_mem mem) {
  assert_single_thread();
  decr_ref_count(mem);
  return PI_SUCCESS;
}

/// Implements a buffer partition in the CUDA backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply implemented
/// as an offset over an existing CUDA allocation.
///
pi_result xrt_piMemBufferPartition(pi_mem parent_buffer, pi_mem_flags flags,
                                   pi_buffer_create_type buffer_create_type,
                                   void *buffer_create_info, pi_mem *memObj) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piMemGetInfo(pi_mem, cl_mem_info, size_t, void *, size_t *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Gets the native CUDA handle of a PI mem object
///
/// \param[in] mem The PI mem to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the PI mem object.
///
/// \return PI_SUCCESS
pi_result xrt_piextMemGetNativeHandle(pi_mem mem,
                                      pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Created a PI mem object from a CUDA mem handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI mem object from.
/// \param[out] mem Set to the PI mem object created from native handle.
///
/// \return TBD
pi_result xrt_piextMemCreateWithNativeHandle(pi_native_handle, pi_mem *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Creates a `pi_queue` object on the XRT backend.
pi_result xrt_piQueueCreate(pi_context context, pi_device device,
                            pi_queue_properties properties, pi_queue *queue) {
  assert_single_thread();
  assert_valid_obj(context);
  assert_valid_obj(device);
  assert(context);
  assert(device);
  // TODO(XRT): : properties not handled

  *queue = ref_counted_base<_pi_queue>::make(context, device);
  return PI_SUCCESS;
}

pi_result xrt_piQueueGetInfo(pi_queue command_queue, pi_queue_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piQueueRetain(pi_queue command_queue) {
  assert_single_thread();
  incr_ref_count(command_queue);
  return PI_SUCCESS;
}

pi_result xrt_piQueueRelease(pi_queue command_queue) {
  assert_single_thread();
  decr_ref_count(command_queue);
  return PI_SUCCESS;
}

/// All commands are executed greedily so this is a noop.
pi_result xrt_piQueueFinish(pi_queue command_queue) {
  assert_single_thread();
  assert_valid_obj(command_queue);
  return PI_SUCCESS;
}

/// All commands are executed greedily so this is a noop.
pi_result xrt_piQueueFlush(pi_queue command_queue) {
  assert_single_thread();
  assert_valid_obj(command_queue);
  return PI_SUCCESS;
}

/// Gets the native CUDA handle of a PI queue object
///
/// \param[in] queue The PI queue to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the PI queue object.
///
/// \return PI_SUCCESS
pi_result xrt_piextQueueGetNativeHandle(pi_queue queue,
                                        pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Created a PI queue object from a CUDA queue handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI queue object from.
/// \param[in] context is the PI context of the queue.
/// \param[out] queue Set to the PI queue object created from native handle.
/// \param ownNativeHandle tells if SYCL RT should assume the ownership of
///        the native handle, if it can.
///
/// \return TBD
pi_result xrt_piextQueueCreateWithNativeHandle(pi_native_handle, pi_context,
                                               pi_queue *,
                                               bool ownNativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                                      pi_bool blocking_write, size_t offset,
                                      size_t size, const void *ptr,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  assert_single_thread();
  assert_valid_obj(command_queue);
  assert_valid_obj(buffer);
  assert_valid_objs(event_wait_list, num_events_in_wait_list);
  assert(ptr && size);
  assert(event);
  *event = ref_counted_base<_pi_event>::make();

  buffer->run_when_mapped(command_queue->device_->get(), [=] {
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
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  assert_single_thread();
  assert_valid_obj(command_queue);
  assert_valid_obj(buffer);
  assert_valid_objs(event_wait_list, num_events_in_wait_list);
  assert(ptr && size);
  assert(event);
  *event = ref_counted_base<_pi_event>::make();

  assert(buffer->is_mapped(command_queue->device_->get()));
  REPRODUCE_MEMCALL(buffer->get_native(), sync, XCL_BO_SYNC_BO_FROM_DEVICE);
  void *adjusted_ptr = ((char *)buffer->mem.mapped_ptr) + offset;
  REPRODUCE_ADD_BUFFER(ptr, size);
  REPRODUCE_CALL((void)std::memcpy, ptr, adjusted_ptr, size);
  return PI_SUCCESS;
}

pi_result xrt_piEventsWait(pi_uint32 num_events, const pi_event *event_list) {
  assert_single_thread();
  assert_valid_objs(event_list, num_events);
  /// For now every operation is executed synchronously this ia a noop

  return PI_SUCCESS;
}

pi_result xrt_piKernelCreate(pi_program program, const char *kernel_name,
                             pi_kernel *kernel) {
  assert_single_thread();
  assert_valid_obj(program);

  /// TODO: XRT error handling
  auto ker = REPRODUCE_CALL(xrt::kernel, program->device_->get(),
                            program->uuid_, kernel_name);
  auto info = REPRODUCE_MEMCALL(program->bin_, get_kernel, kernel_name);
  auto run = REPRODUCE_CALL(xrt::run, ker);
  *kernel = ref_counted_base<_pi_kernel>::make(program, std::move(ker),
                                               std::move(run), std::move(info));
  return PI_SUCCESS;
}

pi_result xrt_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                             size_t arg_size, const void *arg_value) {
  assert_single_thread();
  assert_valid_obj(kernel);
  assert(kernel->info_.get_num_args() > arg_index);
  auto arg_info = kernel->info_.get_arg(arg_index);

  // TODO XRT: analyse if xrt_piKernelSetArg can be called for buffer.
  REPRODUCE_ADD_BUFFER(arg_value, arg_size);
  REPRODUCE_MEMCALL(kernel->run_, set_arg, arg_index, arg_value, arg_size);
  return PI_SUCCESS;
}

pi_result xrt_piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                                      const pi_mem *arg_value) {
  assert_single_thread();
  assert_valid_obj(kernel);
  assert(arg_value);
  _pi_mem *buf = *arg_value;

  buf->map_if_needed(kernel->program_->device_->get());
  REPRODUCE_MEMCALL(kernel->run_, set_arg, arg_index, buf->get_native());

  return PI_SUCCESS;
}

pi_result xrt_piextKernelSetArgSampler(pi_kernel kernel, pi_uint32 arg_index,
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
    pi_queue command_queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  assert_single_thread();
  assert_valid_obj(command_queue);
  assert_valid_obj(kernel);
  assert(work_dim == 1 && *global_work_offset == 0 && *global_work_size == 1 &&
         *local_work_size == 1 && "only support 1 single_task");
  assert(event);
  *event = ref_counted_base<_pi_event>::make();

  REPRODUCE_MEMCALL(kernel->run_, start);
  REPRODUCE_MEMCALL(kernel->run_, wait);

  return PI_SUCCESS;
}

/// \TODO Not implemented
pi_result xrt_piEnqueueNativeKernel(pi_queue, void (*)(void *), void *, size_t,
                                    pi_uint32, const pi_mem *, const void **,
                                    pi_uint32, const pi_event *, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextKernelCreateWithNativeHandle(pi_native_handle, pi_context,
                                                pi_program, bool, pi_kernel *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
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
  assert_single_thread();
  incr_ref_count(mem);
  return PI_SUCCESS;
}

/// Not used as CUDA backend only creates programs from binary.
/// See \ref xrt_piclProgramCreateWithBinary.
///
pi_result xrt_piclProgramCreateWithSource(pi_context, pi_uint32, const char **,
                                          const size_t *, pi_program *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Loads the images from a PI program into a CUmodule that can be
/// used later on to extract functions (kernels).
/// See \ref _pi_program for implementation details.
///
pi_result xrt_piProgramBuild(pi_program program, pi_uint32 num_devices,
                             const pi_device *device_list, const char *options,
                             void (*pfn_notify)(pi_program program,
                                                void *user_data),
                             void *user_data) {
  assert_single_thread();
  assert_valid_obj(program);
  assert(num_devices == 1);
  assert(device_list);
  assert_valid_obj(*device_list);
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);

  return PI_SUCCESS;
}

/// \TODO Not implemented
pi_result xrt_piProgramCreate(pi_context, const void *, size_t, pi_program *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Loads the first xclbin from the list
/// Note: Only supports one device
///
pi_result xrt_piProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *program) {
  assert_single_thread();
  assert_valid_obj(context);
  assert(binaries);
  assert_valid_obj(program);
  assert(num_devices == 1 && "XRT contexts are for a single device");
  assert(device_list);
  assert_valid_obj(*device_list);
  assert((context->devices_[0].value == device_list[0]) &&
         "Mismatch between devices context and passed context when creating "
         "program from binary");

  try {
    /// We assume there is al least 1 valid device
    pi_device dev = device_list[0];
    reproducer() << "// xclbin buffer size=" << lengths[0] << "\n";
    auto xclbin = REPRODUCE_CALL(xrt::xclbin,
                                 reinterpret_cast<const axlf *>(binaries[0]));
    xrt::uuid uuid = REPRODUCE_MEMCALL(dev->get(), load_xclbin, xclbin);

    *program = ref_counted_base<_pi_program>::make(
        context, dev, std::move(xclbin), std::move(uuid));
    return PI_SUCCESS;
  } catch (std::system_error err) {
    sycl::detail::pi::log()
        << "XRT error:" << err.code() << ":" << err.what() << std::endl;
    return PI_ERROR_UNKNOWN;
  }
}

pi_result xrt_piProgramGetInfo(pi_program program, pi_program_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Creates a new PI program object that is the outcome of linking all input
/// programs.
/// \TODO Implement linker options, requires mapping of OpenCL to CUDA
///
pi_result xrt_piProgramLink(pi_context context, pi_uint32 num_devices,
                            const pi_device *device_list, const char *options,
                            pi_uint32 num_input_programs,
                            const pi_program *input_programs,
                            void (*pfn_notify)(pi_program program,
                                               void *user_data),
                            void *user_data, pi_program *ret_program) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Creates a new program that is the outcome of the compilation of the headers
///  and the program.
/// \TODO Implement asynchronous compilation
///
pi_result xrt_piProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piProgramGetBuildInfo(pi_program program, pi_device device,
                                    cl_program_build_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piProgramRetain(pi_program program) {
  assert_single_thread();
  incr_ref_count(program);
  return PI_SUCCESS;
}

/// Decreases the reference count of a pi_program object.
/// When the reference count reaches 0, it unloads the module from
/// the context.
pi_result xrt_piProgramRelease(pi_program program) {
  assert_single_thread();
  decr_ref_count(program);
  return PI_SUCCESS;
}

/// Gets the native CUDA handle of a PI program object
///
/// \param[in] program The PI program to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the PI program object.
///
/// \return TBD
pi_result xrt_piextProgramGetNativeHandle(pi_program program,
                                          pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Created a PI program object from a CUDA program handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI program object from.
/// \param[in] context The PI context of the program.
/// \param[out] program Set to the PI program object created from native handle.
///
/// \return TBD
pi_result xrt_piextProgramCreateWithNativeHandle(pi_native_handle, pi_context,
                                                 bool, pi_program *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piKernelGetInfo(pi_kernel kernel, pi_kernel_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device, pi_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piKernelRetain(pi_kernel kernel) {
  assert_single_thread();
  incr_ref_count(kernel);
  return PI_SUCCESS;
}

pi_result xrt_piKernelRelease(pi_kernel kernel) {
  assert_single_thread();
  decr_ref_count(kernel);
  return PI_SUCCESS;
}

// A NOP for the XRT backend
pi_result xrt_piKernelSetExecInfo(pi_kernel, pi_kernel_exec_info, size_t,
                                  const void *) {
  assert_single_thread();
  return PI_SUCCESS;
}

pi_result xrt_piextKernelSetArgPointer(pi_kernel kernel, pi_uint32 arg_index,
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
  assert_single_thread();
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_INVALID_EVENT;
}

/// Obtain profiling information from PI CUDA events
/// \TODO Untie from OpenCL, timings from CUDA are only elapsed time.
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
  assert_single_thread();
  incr_ref_count(event);
  return PI_SUCCESS;
}

pi_result xrt_piEventRelease(pi_event event) {
  assert_single_thread();
  decr_ref_count(event);
  return PI_SUCCESS;
}

/// Enqueues a wait on the given CUstream for all events.
/// See \ref enqueueEventWait
/// TODO: Add support for multiple streams once the Event class is properly
/// refactored.
///
pi_result xrt_piEnqueueEventsWait(pi_queue command_queue,
                                  pi_uint32 num_events_in_wait_list,
                                  const pi_event *event_wait_list,
                                  pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Enqueues a wait on the given CUstream for all specified events (See
/// \ref enqueueEventWaitWithBarrier.) If the events list is empty, the enqueued
/// wait will wait on all previous events in the queue.
///
/// \param[in] command_queue A valid PI queue.
/// \param[in] num_events_in_wait_list Number of events in event_wait_list.
/// \param[in] event_wait_list Events to wait on.
/// \param[out] event Event for when all events in event_wait_list have finished
/// or, if event_wait_list is empty, when all previous events in the queue have
/// finished.
///
/// \return TBD
pi_result xrt_piEnqueueEventsWaitWithBarrier(pi_queue command_queue,
                                             pi_uint32 num_events_in_wait_list,
                                             const pi_event *event_wait_list,
                                             pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Gets the native CUDA handle of a PI event object
///
/// \param[in] event The PI event to get the native CUDA object of.
/// \param[out] nativeHandle Set to the native handle of the PI event object.
///
/// \return PI_SUCCESS on success. PI_INVALID_EVENT if given a user event.
pi_result xrt_piextEventGetNativeHandle(pi_event event,
                                        pi_native_handle *nativeHandle) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Created a PI event object from a CUDA event handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI event object from.
/// \param[out] event Set to the PI event object created from native handle.
///
/// \return TBD
pi_result xrt_piextEventCreateWithNativeHandle(pi_native_handle, pi_context,
                                               bool, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Creates a PI sampler object
///
/// \param[in] context The context the sampler is created for.
/// \param[in] sampler_properties The properties for the sampler.
/// \param[out] result_sampler Set to the resulting sampler object.
///
/// \return PI_SUCCESS on success. PI_INVALID_VALUE if given an invalid property
///         or if there is multiple of properties from the same category.
pi_result xrt_piSamplerCreate(pi_context context,
                              const pi_sampler_properties *sampler_properties,
                              pi_sampler *result_sampler) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Gets information from a PI sampler object
///
/// \param[in] sampler The sampler to get the information from.
/// \param[in] param_name The name of the information to get.
/// \param[in] param_value_size The size of the param_value.
/// \param[out] param_value Set to information value.
/// \param[out] param_value_size_ret Set to the size of the information value.
///
/// \return PI_SUCCESS on success.
pi_result xrt_piSamplerGetInfo(pi_sampler sampler, cl_sampler_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Retains a PI sampler object, incrementing its reference count.
///
/// \param[in] sampler The sampler to increment the reference count of.
///
/// \return PI_SUCCESS.
pi_result xrt_piSamplerRetain(pi_sampler sampler) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Releases a PI sampler object, decrementing its reference count. If the
/// reference count reaches zero, the sampler object is destroyed.
///
/// \param[in] sampler The sampler to decrement the reference count of.
///
/// \return PI_SUCCESS.
pi_result xrt_piSamplerRelease(pi_sampler sampler) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result enuqeue_rect_copy(bool is_read, pi_queue command_queue, pi_mem buffer,
                            pi_bool blocking, pi_buff_rect_offset buffer_offset,
                            pi_buff_rect_offset host_offset,
                            pi_buff_rect_region region, size_t buffer_row_pitch,
                            size_t buffer_slice_pitch, size_t host_row_pitch,
                            size_t host_slice_pitch, void *ptr,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  assert_single_thread();
  assert_valid_obj(command_queue);
  assert_valid_obj(buffer);
  assert_valid_objs(event_wait_list, num_events_in_wait_list);

  /// TODO add test where the offsets and sizes are not simple
  if (is_read)
    assert(buffer->is_mapped(command_queue->device_->get()));
  buffer->run_when_mapped(
      command_queue->device_->get(),
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
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return enuqeue_rect_copy(
      /*is_read*/ true, command_queue, buffer, blocking_read, buffer_offset,
      host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
      host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

pi_result xrt_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  return enuqeue_rect_copy(
      /*is_read*/ false, command_queue, buffer, blocking_write, buffer_offset,
      host_offset, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
      host_slice_pitch, const_cast<void *>(ptr), num_events_in_wait_list,
      event_wait_list, event);
}

pi_result xrt_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                                     pi_mem dst_buffer, size_t src_offset,
                                     size_t dst_offset, size_t size,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
                                     const void *pattern, size_t pattern_size,
                                     size_t offset, size_t size,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageRead(pi_queue command_queue, pi_mem image,
                                    pi_bool blocking_read, const size_t *origin,
                                    const size_t *region, size_t row_pitch,
                                    size_t slice_pitch, void *ptr,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                                     pi_bool blocking_write,
                                     const size_t *origin, const size_t *region,
                                     size_t input_row_pitch,
                                     size_t input_slice_pitch, const void *ptr,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                                    pi_mem dst_image, const size_t *src_origin,
                                    const size_t *dst_origin,
                                    const size_t *region,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// \TODO Not implemented in CUDA, requires untie from OpenCL
pi_result xrt_piEnqueueMemImageFill(pi_queue, pi_mem, const void *,
                                    const size_t *, const size_t *, pi_uint32,
                                    const pi_event *, pi_event *) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Implements mapping on the host using a BufferRead operation.
/// Mapped pointers are stored in the pi_mem object.
/// If the buffer uses pinned host memory a pointer to that memory is returned
/// and no read operation is done.
/// \TODO Untie types from OpenCL
///
pi_result xrt_piEnqueueMemBufferMap(pi_queue command_queue, pi_mem buffer,
                                    pi_bool blocking_map,
                                    pi_map_flags map_flags, size_t offset,
                                    size_t size,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event, void **ret_map) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// Implements the unmap from the host, using a BufferWrite operation.
/// Requires the mapped pointer to be already registered in the given memobj.
/// If memobj uses pinned host memory, this will not do a write.
///
pi_result xrt_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                void *mapped_ptr,
                                pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// USM: Implements USM Host allocations using CUDA Pinned Memory
///
pi_result xrt_piextUSMHostAlloc(void **result_ptr, pi_context context,
                                pi_usm_mem_properties *properties, size_t size,
                                pi_uint32 alignment) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// USM: Implements USM device allocations using a normal CUDA device pointer
///
pi_result xrt_piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                                  pi_device device,
                                  pi_usm_mem_properties *properties,
                                  size_t size, pi_uint32 alignment) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// USM: Implements USM Shared allocations using CUDA Managed Memory
///
pi_result xrt_piextUSMSharedAlloc(void **result_ptr, pi_context context,
                                  pi_device device,
                                  pi_usm_mem_properties *properties,
                                  size_t size, pi_uint32 alignment) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// USM: Frees the given USM pointer associated with the context.
///
pi_result xrt_piextUSMFree(pi_context context, void *ptr) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                    size_t count,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                    void *dst_ptr, const void *src_ptr,
                                    size_t size,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                      size_t size, pi_usm_migration_flags flags,
                                      pi_uint32 num_events_in_waitlist,
                                      const pi_event *events_waitlist,
                                      pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// USM: memadvise API to govern behavior of automatic migration mechanisms
pi_result xrt_piextUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                       size_t length, pi_mem_advice advice,
                                       pi_event *event) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

/// API to query information about USM allocated pointers
/// Valid Queries:
///   PI_MEM_ALLOC_TYPE returns host/device/shared pi_host_usm value
///   PI_MEM_ALLOC_BASE_PTR returns the base ptr of an allocation if
///                         the queried pointer fell inside an allocation.
///                         Result must fit in void *
///   PI_MEM_ALLOC_SIZE returns how big the queried pointer's
///                     allocation is in bytes. Result is a size_t.
///   PI_MEM_ALLOC_DEVICE returns the pi_device this was allocated against
///
/// \param context is the pi_context
/// \param ptr is the pointer to query
/// \param param_name is the type of query to perform
/// \param param_value_size is the size of the result in bytes
/// \param param_value is the result
/// \param param_value_size_ret is how many bytes were written
pi_result xrt_piextUSMGetMemAllocInfo(pi_context context, const void *ptr,
                                      pi_mem_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  sycl::detail::pi::unimplemented(__PRETTY_FUNCTION__);
}

pi_result xrt_piTearDown(void *) {
  assert_single_thread();
  terminate_xsimk();
  return PI_SUCCESS;
}

const char SupportedVersion[] = _PI_H_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  assert_single_thread();
  int CompareVersions = strcmp(PluginInit->PiVersion, SupportedVersion);
  if (CompareVersions < 0) {
    // PI interface supports lower version of PI.
    // TODO: Take appropriate actions.
    return PI_INVALID_OPERATION;
  }

  std::atexit(terminate_xsimk);

  // PI interface supports higher version or the same version.
  strncpy(PluginInit->PluginVersion, SupportedVersion, 4);

  // Set whole function table to zero to make it easier to detect if
  // functions are not set up below.
  std::memset(&(PluginInit->PiFunctionTable), 0,
              sizeof(PluginInit->PiFunctionTable));

// Forward calls to Xilinx RT (XRT).
#define _PI_CL(pi_api, xrt_api)                                                \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&xrt_api);

  // Platform
  _PI_CL(piPlatformsGet, xrt_piPlatformsGet)
  _PI_CL(piPlatformGetInfo, xrt_piPlatformGetInfo)
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
  _PI_CL(piextContextSetExtendedDeleter, xrt_piextContextSetExtendedDeleter)
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

#pragma clang diagnostic pop
