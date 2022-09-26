// RUN: cgeist %s --function=* -S | FileCheck %s

extern "C" {

double kernel_deriche(int x, float y) {
    ([&y,x]() {
        y *= x;
    })();
    return y;
}

}

// CHECK:   func.func @kernel_deriche(%arg0: i32, %arg1: f32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x!llvm.struct<(memref<?xf32>, i32)>> to memref<?x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %2 = memref.alloca() : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %3 = memref.alloca() : memref<1xf32>
// CHECK-NEXT:     affine.store %arg1, %3[0] : memref<1xf32>
// CHECK-NEXT:     %4 = memref.cast %3 : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:     %5 = "polygeist.memref2pointer"(%2) : (memref<1x!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     llvm.store %4, %6 : !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %7 = llvm.getelementptr %5[0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg0, %7 : !llvm.ptr<i32>
// CHECK-NEXT:     %8 = affine.load %2[0] : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     affine.store %8, %0[0] : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     call @_ZZ14kernel_dericheENK3$_0clEv(%1) : (memref<?x!llvm.struct<(memref<?xf32>, i32)>>) -> ()
// CHECK-NEXT:     %9 = affine.load %3[0] : memref<1xf32>
// CHECK-NEXT:     %10 = arith.extf %9 : f32 to f64
// CHECK-NEXT:     return %10 : f64
// CHECK-NEXT:   }
// CHECK:   func.func private @_ZZ14kernel_dericheENK3$_0clEv(%arg0: memref<?x!llvm.struct<(memref<?xf32>, i32)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %3 = llvm.getelementptr %0[0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<i32>
// CHECK-NEXT:     %5 = arith.sitofp %4 : i32 to f32
// CHECK-NEXT:     %6 = affine.load %2[0] : memref<?xf32>
// CHECK-NEXT:     %7 = arith.mulf %6, %5 : f32
// CHECK-NEXT:     affine.store %7, %2[0] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
