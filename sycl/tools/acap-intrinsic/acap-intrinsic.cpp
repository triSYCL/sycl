
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

void nop5() { ::nop(5); }

uint32_t stream_read32(int stream_idx) { return ::get_ss(stream_idx); }
void stream_write32(int stream_idx, uint32_t val, int tlast) { return ::put_ms(stream_idx, val, tlast); }

}
