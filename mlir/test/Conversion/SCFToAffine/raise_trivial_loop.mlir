// RUN: mlir-opt -raise-scf-to-affine %s | FileCheck %s

// CHECK-LABEL: @trivial_loop
func private @trivial_loop(%arg0: i32, %arg1: i32, %arg3: f64, %arg4: memref<1000x1100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
		%c0 = constant 0 : index
		%c1 = constant 1 : index
    // CHECK: affine.for
    scf.for %arg5 = %c0 to %0 step %c1	{
      // CHECK: affine.for
      scf.for %arg6 = %c0 to %1 step %c1 {
        %3 = load %arg4[%arg5, %arg6] : memref<1000x1100xf64>
        %4 = mulf %3, %arg3 : f64
        store %4, %arg4[%arg5, %arg6] : memref<1000x1100xf64>
      } 
    }
    return
}
