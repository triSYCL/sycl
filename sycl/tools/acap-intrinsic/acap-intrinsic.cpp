
#define __ACAP_RT__
#include "acap-intrinsic.h"

#include <cstring>
#include <array>

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

uint32_t sread(int stream_idx) { return ::get_ss(stream_idx); }
void swrite(int stream_idx, uint32_t val, int tlast) { return ::put_ms(stream_idx, val, tlast); }

void stream_read4(char* out_buffer, int stream_idx) {
  *reinterpret_cast<uint32_t*>(out_buffer) = ::get_ss(stream_idx);
}
void stream_write4(const char* in_buffer, int stream_idx, int tlast) {
  ::put_ms(stream_idx, *reinterpret_cast<uint32_t*>(in_buffer), tlast);
}
void stream_read16(char* out_buffer, int stream_idx) {
  *reinterpret_cast<v4int32*>(out_buffer) = ::getl_wss(stream_idx);
}
void stream_write16(const char* in_buffer, int stream_idx, int tlast) {
  ::put_wms(stream_idx, *reinterpret_cast<v4int32*>(in_buffer), tlast);
}

void cstream_read48(char* out_buffer) {
  *reinterpret_cast<v8acc48*>(out_buffer) = ::get_scd();
}
void cstream_write48(const char* in_buffer) {
  ::put_mcd(*reinterpret_cast<v8acc48*>(in_buffer));
}

}
