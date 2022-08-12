
#include <cstdint>

#define DECLARATIONS(NAME, LhsType, RhsType, OutType) \
extern "C" OutType fixedpt_##NAME##_add(LhsType, RhsType); \
extern "C" OutType fixedpt_##NAME##_sub(LhsType, RhsType); \
extern "C" OutType fixedpt_##NAME##_mul(LhsType, RhsType); \
extern "C" OutType fixedpt_##NAME##_div(LhsType, RhsType);

DECLARATIONS(8_7, int16_t, int16_t, int16_t)
DECLARATIONS(8_7_to_8_5, int16_t, int16_t, int16_t)
DECLARATIONS(6_2_and_8_12_to_8_7, int16_t, int32_t, int16_t)
