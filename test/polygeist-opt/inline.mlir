// RUN: polygeist-opt --inline %s | FileCheck %s

module {
  func.func @eval_cost(%arg0: memref<?x!llvm.struct<"opaque@polygeist@mlir@xyz", (i32, memref<?xf32>)>>, %arg1: i32) -> !llvm.struct<(f32, array<3 x f32>)> {
    %alloca = memref.alloca() : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
    %28 = affine.load %alloca[0] : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
    return %28 : !llvm.struct<(f32, array<3 x f32>)>
  }
  
  func.func @eval_res(%arg0: memref<?x!llvm.struct<"opaque@polygeist@mlir@xyz", (i32, memref<?xf32>)>>) -> !llvm.struct<(f32, array<3 x f32>)>
  {
    %c0_i32 = arith.constant 0 : i32
    %1 = func.call @eval_cost(%arg0, %c0_i32) : (memref<?x!llvm.struct<"opaque@polygeist@mlir@xyz", (i32, memref<?xf32>)>>, i32) -> !llvm.struct<(f32, array<3 x f32>)>
    return %1 : !llvm.struct<(f32, array<3 x f32>)>
  }
// CHECK: module {
// CHECK:   func.func @eval_cost(%arg0: memref<?x!llvm.struct<"opaque@polygeist@mlir@xyz", (i32, memref<?xf32>)>>, %arg1: i32) -> !llvm.struct<(f32, array<3 x f32>)> {
// CHECK:     %[[V0:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:     %[[T0:.*]] = affine.load %[[V0:.*]][0] : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:     return %[[T0:.*]] : !llvm.struct<(f32, array<3 x f32>)>
// CHECK:   }
// CHECK:   func.func @eval_res(%arg0: memref<?x!llvm.struct<"opaque@polygeist@mlir@xyz", (i32, memref<?xf32>)>>) -> !llvm.struct<(f32, array<3 x f32>)> {
// CHECK:     %[[V0:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:     %[[T0:.*]] = affine.load %[[V0:.*]][0] : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:     return %[[T0:.*]] : !llvm.struct<(f32, array<3 x f32>)>
// CHECK:   }
// CHECK: }
}