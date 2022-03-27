// RUN: polygeist-opt --affine-cfg --split-input-file %s | FileCheck %s
module {
  func @_Z7runTestiPPc(%arg0: index, %arg2: memref<?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %1 = arith.addi %arg0, %c1 : index
    affine.for %arg3 = 0 to 2 {
      %2 = arith.muli %arg3, %1 : index
      affine.for %arg4 = 0 to 2 {
        %3 = arith.addi %2, %arg4 : index
        memref.store %c0_i32, %arg2[%3] : memref<?xi32>
      }
    }
    return
  }

}


// CHECK:   func @_Z7runTestiPPc(%arg0: index, %arg1: memref<?xi32>) {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     affine.for %arg2 = 0 to 2 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 2 {
// CHECK-NEXT:         affine.store %c0_i32, %arg1[%arg3 + %arg2 * (symbol(%arg0) + 1)] : memref<?xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----
module {
func @kernel_nussinov(%arg0: i32, %arg2: memref<i32>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %c1_i32 = arith.constant 1 : i32
  %c59 = arith.constant 59 : index
  %c100_i32 = arith.constant 100 : i32
  affine.for %arg3 = 0 to 60 {
    %0 = arith.subi %c59, %arg3 : index
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.cmpi slt, %1, %c100_i32 : i32
    scf.if %2 {
      affine.store %arg0, %arg2[] : memref<i32>
    }
  }
  return
}
}

// CHECK: #set = affine_set<(d0) : (d0 + 40 >= 0)>
// CHECK:   func @kernel_nussinov(%arg0: i32, %arg1: memref<i32>) {
// CHECK-NEXT:     affine.for %arg2 = 0 to 60 {
// CHECK-NEXT:       affine.if #set(%arg2) {
// CHECK-NEXT:         affine.store %arg0, %arg1[] : memref<i32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// -----

module {
  func private @run()

  func @minif(%arg4: i32, %arg5 : i32, %arg10 : index) { 
    %c0_i32 = arith.constant 0 : i32
      
    affine.for %i = 0 to 10 {
      %70 = arith.index_cast %arg10 : index to i32
      %71 = arith.muli %70, %arg5 : i32
      %73 = arith.divui %71, %arg5 : i32
      %75 = arith.muli %73, %arg5 : i32
      %79 = arith.subi %arg4, %75 : i32
      %81 = arith.cmpi sle, %arg5, %79 : i32
      %83 = arith.select %81, %arg5, %79 : i32
      %92 = arith.cmpi slt, %c0_i32, %83 : i32
      scf.if %92 {
        call @run() : () -> ()
        scf.yield
      }
    }
    return
  }
}

// CHECK: #set = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK:   func @minif(%arg0: i32, %arg1: i32, %arg2: index) {
// CHECK-NEXT:     %0 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:     %1 = arith.muli %0, %arg1 : i32
// CHECK-NEXT:     %2 = arith.divui %1, %arg1 : i32
// CHECK-NEXT:     %3 = arith.muli %2, %arg1 : i32
// CHECK-NEXT:     %4 = arith.subi %arg0, %3 : i32
// CHECK-NEXT:     %5 = arith.cmpi sle, %arg1, %4 : i32
// CHECK-NEXT:     %6 = arith.select %5, %arg1, %4 : i32
// CHECK-NEXT:     %7 = arith.index_cast %6 : i32 to index
// CHECK-NEXT:     affine.for %arg3 = 0 to 10 {
// CHECK-NEXT:       affine.if #set()[%7] {
// CHECK-NEXT:         call @run() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
