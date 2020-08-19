// RUN: %clang_cc1 -fsycl -fsycl-is-device -fcxx-exceptions -Wno-return-type -verify -fsyntax-only -std=c++20 -Werror=vla %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

// expected-note@+1{{function implemented using recursion declared here}}
constexpr int constexpr_recurse1(int n);

// expected-note@+1 3{{function implemented using recursion declared here}}
constexpr int constexpr_recurse(int n) {
  if (n)
    // expected-error@+1{{SYCL kernel cannot call a recursive function}}
    return constexpr_recurse1(n - 1);
  return 103;
}

constexpr int constexpr_recurse1(int n) {
  // expected-error@+1{{SYCL kernel cannot call a recursive function}}
  return constexpr_recurse(n) + 1;
}

template <int I>
void bar() {}

template <int... args>
void bar2() {}

enum class SomeE {
  Value = constexpr_recurse(5)
};

struct ConditionallyExplicitCtor {
  explicit(constexpr_recurse(5) == 103) ConditionallyExplicitCtor(int i) {}
};

void conditionally_noexcept() noexcept(constexpr_recurse(5)) {}

// All of the uses of constexpr_recurse here are forced constant expressions, so
// they should not diagnose.
void constexpr_recurse_test() {
  constexpr int i = constexpr_recurse(1);
  bar<constexpr_recurse(2)>();
  bar2<1, 2, constexpr_recurse(2)>();
  static_assert(constexpr_recurse(2) == 105, "");

  int j;
  switch (105) {
  case constexpr_recurse(2):
    // expected-error@+1{{SYCL kernel cannot call a recursive function}}
    j = constexpr_recurse(5);
    break;
  }

  SomeE e = SomeE::Value;

  int ce_array[constexpr_recurse(5)];

  conditionally_noexcept();

  if constexpr ((bool)SomeE::Value) {
  }

  ConditionallyExplicitCtor c(1);
}

void constexpr_recurse_test_err() {
  // expected-error@+1{{SYCL kernel cannot call a recursive function}}
  int i = constexpr_recurse(1);
}

int main() {
  kernel_single_task<class fake_kernel>([]() { constexpr_recurse_test(); });
  kernel_single_task<class fake_kernel>([]() { constexpr_recurse_test_err(); });
}
