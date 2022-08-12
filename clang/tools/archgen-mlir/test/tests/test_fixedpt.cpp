
// RUN: archgen-opt %S/Inputs/fixedpt.mlir --convert-fixedpt-to-arith -canonicalize -cse --convert-arith-to-llvm --convert-func-to-llvm | mlir-translate --mlir-to-llvmir -o - | llc -filetype=obj -o %t.fixed.o
// RUN: %clangxx -g -std=c++17 %s %t.fixed.o -o %t.out
// RUN: %t.out | FileCheck %s

// It might be simpler to just extand the MLIR generator to emit some fixedpt operations

#include <iostream>
#include "Inputs/fixedpt.h"

/// Divition to prevent executing div by 0
struct noDivRAII {
  static inline bool noDiv = false;
  noDivRAII() {noDiv = true;}
  ~noDivRAII() {noDiv = false;}
};

#define PRINTER(POSTFIX, OUTMASK)                                              \
  void print_##POSTFIX(int lhs, int rhs) {                                     \
    uint64_t outMask = (OUTMASK);                                              \
    std::cout << std::hex << "add 0x"                                          \
              << (fixedpt_##POSTFIX##_add(lhs, rhs) & outMask);                \
    std::cout << std::hex << " mul 0x"                                         \
              << (fixedpt_##POSTFIX##_mul(lhs, rhs) & outMask);                \
    std::cout << std::hex << " sub 0x"                                         \
              << (fixedpt_##POSTFIX##_sub(lhs, rhs) & outMask);                \
    if (!noDivRAII::noDiv)                                                     \
      std::cout << std::hex << " div 0x"                                       \
                << (fixedpt_##POSTFIX##_div(lhs, rhs) & outMask);              \
    std::cout << std::hex << std::endl;                                        \
  }

PRINTER(8_7, 0xffff)
PRINTER(8_7_to_8_5, 0x3fff)
PRINTER(6_2_and_8_12_to_8_7, 0xffff)

int main() {
  {
    noDivRAII nodiv;
    print_8_7(0, 0);
    // CHECK: add 0x0 mul 0x0 sub 0x0
    print_8_7(1, 0);
    // CHECK: add 0x1 mul 0x0 sub 0x1
    print_8_7(0xffff, 0);
    // CHECK: add 0xffff mul 0x0 sub 0xffff
  }
  print_8_7(0x7fff, 1);
  // CHECK: add 0x8000 mul 0xff sub 0x7ffe
  print_8_7(0, 1);
  // CHECK: add 0x1 mul 0x0 sub 0xffff div 0x0
  print_8_7(1, 1);
  // CHECK: add 0x2 mul 0x0 sub 0x0 div 0x80
  print_8_7(2, 1);
  // CHECK: add 0x3 mul 0x0 sub 0x1 div 0x100
  print_8_7(2, 2);
  // CHECK: add 0x4 mul 0x0 sub 0x0 div 0x80
  print_8_7(1 << 7, 1 << 7);
  // CHECK: add 0x100 mul 0x80 sub 0x0 div 0x80
  print_8_7(2 << 7, 1 << 7);
  // CHECK: add 0x180 mul 0x100 sub 0x80 div 0x100
  print_8_7(2 << 7, 2 << 7);
  // CHECK: add 0x200 mul 0x200 sub 0x0 div 0x80
  /// The division in the next one seems wrong to me
  print_8_7(0xffff, 1 << 7);
  // CHECK: add 0x7f mul 0xffff sub 0xff7f
}
