// RUN: polygeist-opt --canonicalize --allow-unregistered-dialect --split-input-file %s | FileCheck %s

#set0 = affine_set<(d0) : (-d0 == 0)>
#set1 = affine_set<(d0) : (d0 == 0)>
module {
  func.func @bpnn_train_cuda() {
      affine.parallel (%arg7) = (0) to (16) {
        "test.pre"() : () -> ()
        affine.if #set0(%arg7) {
          %a = "test.create"() : () -> i32
          "test.use"(%a) : (i32) -> ()
        }
    }
    return
  }
  func.func @bpnn_train_cuda1() {
      affine.parallel (%arg7) = (0) to (16) {
        "test.pre"() : () -> ()
        affine.if #set1(%arg7) {
          %a = "test.create"() : () -> i32
          "test.use"(%a) : (i32) -> ()
        }
    }
    return
  }
  func.func @bpnn_train_cuda2() {
      affine.parallel (%arg7) = (0) to (16) {
        %a = "test.create"() : () -> i32
        affine.if #set1(%arg7) {
          "test.use"(%a) : (i32) -> ()
        }
    }
    return
  }
}

// CHECK:   func.func @bpnn_train_cuda() {
// CHECK-NEXT:     affine.parallel (%[[arg0:.+]]) = (0) to (16) {
// CHECK-NEXT:       "test.pre"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V0:.+]] = "test.create"() : () -> i32
// CHECK-NEXT:     "test.use"(%[[V0]]) : (i32) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @bpnn_train_cuda1() {
// CHECK-NEXT:     affine.parallel (%[[arg0:.+]]) = (0) to (16) {
// CHECK-NEXT:       "test.pre"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V0:.+]] = "test.create"() : () -> i32
// CHECK-NEXT:     "test.use"(%[[V0]]) : (i32) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @bpnn_train_cuda2() {
// CHECK-NEXT:     affine.parallel (%[[arg0:.+]]) = (0) to (16) {
// CHECK-NEXT:       %[[V0:.+]] = "test.create"() : () -> i32
// CHECK-NEXT:       affine.if #set(%[[arg0]]) {
// CHECK-NEXT:         "test.use"(%[[V0]]) : (i32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
