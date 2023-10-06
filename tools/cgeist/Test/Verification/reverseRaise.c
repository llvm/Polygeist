// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

void use(int i);

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int start, int end) {
  for (int i = end; i >= start; i--) {
    use(i);
  }
}

// CHECK-LABEL:   func.func @kernel_correlation(
// CHECK-SAME:                                  %[[VAL_0:[A-Za-z0-9_]*]]: i32,
// CHECK-SAME:                                  %[[VAL_1:[A-Za-z0-9_]*]]: i32)  
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           affine.for %[[VAL_4:[A-Za-z0-9_]*]] = %[[VAL_3]] to #map(){{\[}}%[[VAL_2]]] {
// CHECK:             %[[VAL_5:[A-Za-z0-9_]*]] = arith.subi %[[VAL_4]], %[[VAL_3]] : index
// CHECK:             %[[VAL_6:[A-Za-z0-9_]*]] = arith.subi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:             %[[VAL_7:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:             func.call @use(%[[VAL_7]]) : (i32) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @use(i32) 

