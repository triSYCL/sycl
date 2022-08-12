module {
  func.func public @fixedpt_8_7_add(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.add"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
  func.func public @fixedpt_8_7_sub(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.sub"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
  func.func public @fixedpt_8_7_mul(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.mul"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
  func.func public @fixedpt_8_7_div(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.div"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }

  func.func public @fixedpt_8_7_to_8_5_add(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -5, "signed">) {
    %result = "fixedpt.add"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -5, "signed">
    return %result : !fixedpt.fixedPt<8, -5, "signed">
  }
  func.func public @fixedpt_8_7_to_8_5_sub(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -5, "signed">) {
    %result = "fixedpt.sub"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -5, "signed">
    return %result : !fixedpt.fixedPt<8, -5, "signed">
  }
  func.func public @fixedpt_8_7_to_8_5_mul(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -5, "signed">) {
    %result = "fixedpt.mul"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -5, "signed">
    return %result : !fixedpt.fixedPt<8, -5, "signed">
  }
  func.func public @fixedpt_8_7_to_8_5_div(%lhs: !fixedpt.fixedPt<8, -7, "signed">, %rhs: !fixedpt.fixedPt<8, -7, "signed">) -> (!fixedpt.fixedPt<8, -5, "signed">) {
    %result = "fixedpt.div"(%lhs, %rhs) : (!fixedpt.fixedPt<8, -7, "signed">, !fixedpt.fixedPt<8, -7, "signed">) -> !fixedpt.fixedPt<8, -5, "signed">
    return %result : !fixedpt.fixedPt<8, -5, "signed">
  }


  func.func public @fixedpt_6_2_and_8_12_to_8_7_add(%lhs: !fixedpt.fixedPt<6, -2, "signed">, %rhs: !fixedpt.fixedPt<8, -12, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.add"(%lhs, %rhs) : (!fixedpt.fixedPt<6, -2, "signed">, !fixedpt.fixedPt<8, -12, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
  func.func public @fixedpt_6_2_and_8_12_to_8_7_sub(%lhs: !fixedpt.fixedPt<6, -2, "signed">, %rhs: !fixedpt.fixedPt<8, -12, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.sub"(%lhs, %rhs) : (!fixedpt.fixedPt<6, -2, "signed">, !fixedpt.fixedPt<8, -12, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
  func.func public @fixedpt_6_2_and_8_12_to_8_7_mul(%lhs: !fixedpt.fixedPt<6, -2, "signed">, %rhs: !fixedpt.fixedPt<8, -12, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.mul"(%lhs, %rhs) : (!fixedpt.fixedPt<6, -2, "signed">, !fixedpt.fixedPt<8, -12, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
  func.func public @fixedpt_6_2_and_8_12_to_8_7_div(%lhs: !fixedpt.fixedPt<6, -2, "signed">, %rhs: !fixedpt.fixedPt<8, -12, "signed">) -> (!fixedpt.fixedPt<8, -7, "signed">) {
    %result = "fixedpt.div"(%lhs, %rhs) : (!fixedpt.fixedPt<6, -2, "signed">, !fixedpt.fixedPt<8, -12, "signed">) -> !fixedpt.fixedPt<8, -7, "signed">
    return %result : !fixedpt.fixedPt<8, -7, "signed">
  }
}
