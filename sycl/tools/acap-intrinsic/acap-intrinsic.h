#ifndef __ACAP_INTRINSIC_HPP__
#define __ACAP_INTRINSIC_HPP__

/// __SYCL_DEVICE_ONLY__ means we are on device code in SYCL
/// __ACAP_RT__ means we are compiling the device library in chess
/// neither __SYCL_DEVICE_ONLY__ nor __ACAP_RT__ means we are compiling for the
/// host.

#if defined(__SYCL_DEVICE_ONLY__) || defined(__ACAP_RT__)
/// When compiling for device the definition is in the library
#define DECL_POSTFIX ;
#if defined(__SYCL_DEVICE_ONLY__)
/// When compiling for SYCL we need to mark function external or the compiler
/// will complain about undefined function.
#define DECL_PREFIX SYCL_EXTERNAL
#else
#define DECL_PREFIX
#endif
#else
#include <cassert>
/// When compiling for the host the functions need to compile because they will
/// be compiled even they will not be executed.
#define DECL_POSTFIX                                                           \
  {                                                                            \
    bool b = false;                                                            \
    (void)b;                                                                   \
    assert(b && "this should only be called on device");                       \
    __builtin_unreachable();                                                   \
  }
#define DECL_PREFIX
#endif

#include <stdint.h>

namespace acap_intr {

DECL_PREFIX __attribute__((const)) int get_coreid(void) DECL_POSTFIX

DECL_PREFIX void memory_fence(void) DECL_POSTFIX
DECL_PREFIX void separator_scheduler(void) DECL_POSTFIX

DECL_PREFIX void acquire(unsigned id, unsigned val) DECL_POSTFIX
DECL_PREFIX void release(unsigned id, unsigned val) DECL_POSTFIX
DECL_PREFIX void acquire(unsigned id) DECL_POSTFIX
DECL_PREFIX void release(unsigned id) DECL_POSTFIX

DECL_PREFIX void core_done() DECL_POSTFIX

DECL_PREFIX void nop5() DECL_POSTFIX

/// Read or write a 32 bit value from/to a stream, using the simpler API.
/// TODO maybe this should be moved to hardware.hpp
DECL_PREFIX uint32_t sread(int stream_idx) DECL_POSTFIX
DECL_PREFIX void swrite(int stream_idx, uint32_t val, int tlast = false) DECL_POSTFIX

/// These funtions assume the buffer is large enough for the read or the write.
/// It is the caller responsability to make sure that the provided pointer are valid for all accessed bytes.
/// The number at the end of the function name indicate how many bytes after the pointer will be accessed.

DECL_PREFIX void stream_read4(char* out_buffer, int stream_idx) DECL_POSTFIX
DECL_PREFIX void stream_write4(const char* in_buffer, int stream_idx, int tlast = false) DECL_POSTFIX
DECL_PREFIX void stream_read16(char* out_buffer, int stream_idx) DECL_POSTFIX
DECL_PREFIX void stream_write16(const char* in_buffer, int stream_idx, int tlast = false) DECL_POSTFIX
/// These function will access cascade stream instead of normal streams.
DECL_PREFIX void cstream_read48(char* out_buffer) DECL_POSTFIX
DECL_PREFIX void cstream_write48(const char* in_buffer) DECL_POSTFIX

}

#endif
