// RUN: mlir-clang %s --function=* -S | FileCheck %s

void histo_kernel(int i);

int runHisto(int totalNum, int partialNum) {
    for(int i = 0; i < totalNum; i+=partialNum*2)
    {
        histo_kernel(i);
    }
    return 0;
}

// CHECK:   func @runHisto(%arg0: i32, %arg1: i32) -> i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = arith.muli %arg1, %c2_i32 : i32
// CHECK-NEXT:     %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     %2 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = scf.for %arg2 = %c0 to %1 step %2 iter_args(%arg3 = %c0_i32) -> (i32) {
// CHECK-NEXT:       call @histo_kernel(%arg3) : (i32) -> ()
// CHECK-NEXT:       %4 = arith.addi %arg3, %0 : i32
// CHECK-NEXT:       scf.yield %4 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
