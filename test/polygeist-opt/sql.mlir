// RUN: polygeist-opt %s | FileCheck %s

module  {
  func.func private @run() -> i32 {
    %c0 = arith.constant 0 : index
    %q = "sql.select"() {column = ["data"], table = "mytable"} : () -> index
    %h = "sql.execute"(%q) : (index) -> index
    %res = "sql.get_result"(%h, %c0) {column = "data"} : (index, index) -> i32
    return %res : i32
  }
}