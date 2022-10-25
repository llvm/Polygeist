// RUN: cgeist %s --function=* -S | FileCheck %s

extern "C" {

double kernel_deriche(int x, float y) {
    ([&y,x]() {
        y *= x;
    })();
    return y;
}

}

// CHECK:   func.func @kernel_deriche(%[[arg0:.+]]: i32, %[[arg1:.+]]: f32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x!llvm.struct<(memref<?xf32>, i32)>> to memref<?x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %[[V2:.+]] = memref.alloca() : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %[[V3:.+]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:     affine.store %[[arg1]], %[[V3]][0] : memref<1xf32>
// CHECK-NEXT:     %[[V4:.+]] = memref.cast %[[V3]] : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:     %[[V5:.+]] = "polygeist.memref2pointer"(%[[V2]]) : (memref<1x!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %[[V6:.+]] = llvm.getelementptr %[[V5]][0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     llvm.store %[[V4]], %[[V6]] : !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %[[V7:.+]] = llvm.getelementptr %[[V5]][0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %[[arg0]], %[[V7]] : !llvm.ptr<i32>
// CHECK-NEXT:     %[[V8:.+]] = affine.load %[[V2]][0] : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     affine.store %[[V8]], %[[V0]][0] : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     call @_ZZ14kernel_dericheENK3$_0clEv(%[[V1]]) : (memref<?x!llvm.struct<(memref<?xf32>, i32)>>) -> ()
// CHECK-NEXT:     %[[V9:.+]] = affine.load %[[V3]][0] : memref<1xf32>
// CHECK-NEXT:     %[[V10:.+]] = arith.extf %[[V9]] : f32 to f64
// CHECK-NEXT:     return %[[V10]] : f64
// CHECK-NEXT:   }
// CHECK:   func.func private @_ZZ14kernel_dericheENK3$_0clEv(%[[arg0:.+]]: memref<?x!llvm.struct<(memref<?xf32>, i32)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.load %[[V1]] : !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %[[V3:.+]] = llvm.getelementptr %[[V0]][0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[V4:.+]] = llvm.load %[[V3]] : !llvm.ptr<i32>
// CHECK-NEXT:     %[[V5:.+]] = arith.sitofp %[[V4]] : i32 to f32
// CHECK-NEXT:     %[[V6:.+]] = affine.load %[[V2]][0] : memref<?xf32>
// CHECK-NEXT:     %[[V7:.+]] = arith.mulf %[[V6]], %[[V5]] : f32
// CHECK-NEXT:     affine.store %[[V7]], %[[V2]][0] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
