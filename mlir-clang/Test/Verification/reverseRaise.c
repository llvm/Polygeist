// RUN: mlir-clang %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s

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

// CHECK:   #map = affine_map<()[s0] -> (s0 + 1)>
// CHECK:   func @kernel_correlation(%arg0: i32, %arg1: i32)
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg2 = %1 to #map()[%0] {
// CHECK-NEXT:       %2 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %3 = arith.subi %2, %arg0 : i32
// CHECK-NEXT:       %4 = arith.subi %arg1, %3 : i32
// CHECK-NEXT:       call @use(%4) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
