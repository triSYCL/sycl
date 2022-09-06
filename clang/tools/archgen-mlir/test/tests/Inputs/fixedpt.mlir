module {
  func.func public @fixedpt_8_7_add(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.add"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
  func.func public @fixedpt_8_7_sub(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.sub"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
  func.func public @fixedpt_8_7_mul(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.mul"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
  func.func public @fixedpt_8_7_div(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.div"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }

  func.func public @fixedpt_8_7_to_8_5_add(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -5, "s">) {
    %result = "fixedpt.add"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -5, "s">
    return %result : !fixedpt.fixedPt<8, -5, "s">
  }
  func.func public @fixedpt_8_7_to_8_5_sub(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -5, "s">) {
    %result = "fixedpt.sub"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -5, "s">
    return %result : !fixedpt.fixedPt<8, -5, "s">
  }
  func.func public @fixedpt_8_7_to_8_5_mul(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -5, "s">) {
    %result = "fixedpt.mul"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -5, "s">
    return %result : !fixedpt.fixedPt<8, -5, "s">
  }
  func.func public @fixedpt_8_7_to_8_5_div(%lhs: !fixedpt.fixedPt<8, -7, "s">, %rhs: !fixedpt.fixedPt<8, -7, "s">) -> (!fixedpt.fixedPt<8, -5, "s">) {
    %result = "fixedpt.div"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<8, -7, "s">, !fixedpt.fixedPt<8, -7, "s">) -> !fixedpt.fixedPt<8, -5, "s">
    return %result : !fixedpt.fixedPt<8, -5, "s">
  }


  func.func public @fixedpt_6_2_and_8_12_to_8_7_add(%lhs: !fixedpt.fixedPt<6, -2, "s">, %rhs: !fixedpt.fixedPt<8, -12, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.add"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<6, -2, "s">, !fixedpt.fixedPt<8, -12, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
  func.func public @fixedpt_6_2_and_8_12_to_8_7_sub(%lhs: !fixedpt.fixedPt<6, -2, "s">, %rhs: !fixedpt.fixedPt<8, -12, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.sub"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<6, -2, "s">, !fixedpt.fixedPt<8, -12, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
  func.func public @fixedpt_6_2_and_8_12_to_8_7_mul(%lhs: !fixedpt.fixedPt<6, -2, "s">, %rhs: !fixedpt.fixedPt<8, -12, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.mul"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<6, -2, "s">, !fixedpt.fixedPt<8, -12, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
  func.func public @fixedpt_6_2_and_8_12_to_8_7_div(%lhs: !fixedpt.fixedPt<6, -2, "s">, %rhs: !fixedpt.fixedPt<8, -12, "s">) -> (!fixedpt.fixedPt<8, -7, "s">) {
    %result = "fixedpt.div"(%lhs, %rhs) {rounding = 0 : i32} : (!fixedpt.fixedPt<6, -2, "s">, !fixedpt.fixedPt<8, -12, "s">) -> !fixedpt.fixedPt<8, -7, "s">
    return %result : !fixedpt.fixedPt<8, -7, "s">
  }
}
