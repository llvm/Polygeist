// RUN: mlir-clang %s --function=foo -S | FileCheck %s

int foo(int t) {
  switch (t) {
  }
  return t;
}

// TODO the select should be canonicalized better
// CHECK:   func @foo(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     return %arg0 : i32
// CHECK-NEXT:   }
