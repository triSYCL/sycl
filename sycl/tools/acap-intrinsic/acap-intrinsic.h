
#ifdef __SYCL_ACAP_INTRINSIC_H__
#define __SYCL_ACAP_INTRINSIC_H__

#if defined(__SYCL_DEVICE_ONLY__) || defined(__ACAP_RT__)
#define DECL_POSTFIX ;
#else
#include <cassert>
#define DECL_POSTFIX                                                           \
  {                                                                            \
    bool b = false;                                                            \
    (void)b;                                                                   \
    assert(b && "this should only be called on device");                       \
    __builtin_unreachable();                                                   \
  }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define DECL_PREFIX SYCL_EXTERNAL
#else
#define DECL_PREFIX inline
#endif

namespace acap_intr {

DECL_PREFIX int get_coreid(void) DECL_POSTFIX

}

#endif
