/// __SYCL_DEVICE_ONLY__ means we are on device code in sycl
/// __ACAP_RT__ means we are compilling the device library
/// neither __SYCL_DEVICE_ONLY__ nor __ACAP_RT__ means we are compilling for the
/// host.

#if defined(__SYCL_DEVICE_ONLY__) || defined(__ACAP_RT__)
/// When compilling for device the definition is in the library
#define DECL_POSTFIX ;
#if defined(__SYCL_DEVICE_ONLY__)
/// When compilling for sycl we need to mark function external or the compiler
/// will complain about undefiend function.
#define DECL_PREFIX SYCL_EXTERNAL
#else
#define DECL_PREFIX
#endif
#else
#include <cassert>
/// when compiling for the host the functions need to compile because they will
/// be compiled but they shouldn't be executed.
#define DECL_POSTFIX                                                           \
  {                                                                            \
    bool b = false;                                                            \
    (void)b;                                                                   \
    assert(b && "this should only be called on device");                       \
    __builtin_unreachable();                                                   \
  }
/// When compilling for the host functions need to be inline because they may
/// colide
#define DECL_PREFIX inline
#endif

namespace acap_intr {

DECL_PREFIX int get_coreid(void) DECL_POSTFIX

DECL_PREFIX void memory_fence(void) DECL_POSTFIX

DECL_PREFIX void acquire(unsigned id, unsigned val) DECL_POSTFIX
DECL_PREFIX void release(unsigned id, unsigned val) DECL_POSTFIX
DECL_PREFIX void acquire(unsigned id) DECL_POSTFIX
DECL_PREFIX void release(unsigned id) DECL_POSTFIX

}
