#ifndef ARCHGEN_H_
#define ARCHGEN_H_

namespace std {
using size_t = __SIZE_TYPE__;
}

#define ARCHGEN_MLIR_ATTR(STR) __attribute__((annotate("archgen_mlir_" #STR)))

namespace archgenlib {

template <int msb, int lsb, typename SignTy> struct FixedFormat {
  static constexpr int width = msb - lsb + 1;
};

template <typename FormatTy> struct FixedNumber {
  _BitInt(FormatTy::width) value;
};

template <typename T1, typename T2> struct AddOp {};

template <typename T1> struct SinOp {};

struct PiOp {};

template <typename T, auto Val> struct ConstantImpl {
  using dimention_t = T;
  static constexpr auto value = Val;
};

template <typename Inner> struct Constant {};

template <typename T, std::size_t ID> struct Variable {};

namespace detail {

struct ToBeFolded {};

template <typename T = ToBeFolded>
ARCHGEN_MLIR_ATTR(generic_op)
ARCHGEN_MLIR_ATTR(emit_as_mlir) T generic_op(...);

template <typename ET> struct evaluatorImpl {};

template <typename InnerET> struct evaluatorImpl<SinOp<InnerET>> {
  static ARCHGEN_MLIR_ATTR(emit_as_mlir) ToBeFolded evaluate() {
    return generic_op<ToBeFolded>("sin", evaluatorImpl<InnerET>::evaluate());
  }
};

template <typename LeftET, typename RightET>
struct evaluatorImpl<AddOp<LeftET, RightET>> {
  static ARCHGEN_MLIR_ATTR(emit_as_mlir) ToBeFolded evaluate() {
    return generic_op<ToBeFolded>("add", evaluatorImpl<LeftET>::evaluate(),
                                  evaluatorImpl<RightET>::evaluate());
  }
};

template <> struct evaluatorImpl<PiOp> {
  static ARCHGEN_MLIR_ATTR(emit_as_mlir) ToBeFolded evaluate() {
    return generic_op<ToBeFolded>("pi");
  }
};

template <typename NumTy, std::size_t ID>
struct evaluatorImpl<::archgenlib::Variable<NumTy, ID>> {
  static ARCHGEN_MLIR_ATTR(emit_as_mlir) ToBeFolded evaluate() {
    return generic_op<ToBeFolded>("variable",
                                  generic_op<NumTy>("parameter", ID));
  }
};

template <typename FixedConstTy>
struct evaluatorImpl<::archgenlib::Constant<FixedConstTy>> {
  static ARCHGEN_MLIR_ATTR(emit_as_mlir) ToBeFolded evaluate() {
    return generic_op<ToBeFolded>(
        "constant",
        ::archgenlib::FixedNumber<typename FixedConstTy::dimension_t>{
            FixedConstTy::value});
  }
};

template <typename T, typename ET, typename... Ts>
ARCHGEN_MLIR_ATTR(emit_as_mlir)
ARCHGEN_MLIR_ATTR(top_level) T evaluateImpl(Ts... ts) {
  return detail::generic_op<T>("evaluate",
                               detail::evaluatorImpl<ET>::evaluate());
}

} // namespace detail
} // namespace archgenlib

#undef ARCHGEN_MLIR_ATTR
#endif
