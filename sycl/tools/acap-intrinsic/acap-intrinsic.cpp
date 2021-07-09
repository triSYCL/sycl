
#define __ACAP_RT__
#include "acap-intrinsic.h"

namespace acap_intr {

int get_coreid() { return ::get_coreid(); }

void memory_fence() { ::chess_memory_fence();  }
void separator_scheduler() { ::chess_separator_scheduler(); }

void acquire(unsigned id, unsigned val) { return ::acquire(id, val); }
void release(unsigned id, unsigned val) { return ::release(id, val); }
void acquire(unsigned id) { return ::acquire(id); }
void release(unsigned id) { return ::release(id); }

void core_done() { ::done(); }

}
