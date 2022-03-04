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
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.muli %arg1, %c2_i32 : i32
// CHECK-NEXT:     %2 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     %3 = arith.muli %0, %c2 : index
// CHECK-NEXT:     scf.for %arg2 = %c0 to %2 step %3 {
// CHECK-NEXT:       %4 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %5 = arith.divui %4, %1 : i32
// CHECK-NEXT:       %6 = arith.muli %5, %1 : i32
// CHECK-NEXT:       call @histo_kernel(%6) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
