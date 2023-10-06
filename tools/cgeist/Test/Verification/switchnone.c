// RUN: cgeist %s --function=foo -S | FileCheck %s

int foo(int t) {
  switch (t) {
  }
  return t;
}

// CHECK:   func @foo(%[[arg0:.+]]: i32) -> i32  
// CHECK-NEXT:     return %[[arg0]] : i32
// CHECK-NEXT:   }
