// RUN: polygeist-opt -convert-polygeist-to-llvm %s | FileCheck %s

memref.global constant @glob_1d : memref<42xf32> = dense<10.1>

memref.global @glob_2d : memref<10x5xf32> = dense<4.2>

func.func @global_1d(%arg1: index) -> f32 {
  %1 = memref.get_global @glob_1d : memref<42xf32>
  %2 = memref.load %1[%arg1] : memref<42xf32>
  return %2 : f32
}

func.func @global_2d(%arg0: index, %arg1: index, %value: f32) {
  %1 = memref.get_global @glob_2d : memref<10x5xf32>
  memref.store %value, %1[%arg0, %arg1] : memref<10x5xf32>
  return
}

func.func @alloc_0d() -> memref<f32> {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

func.func @alloc_1d_dynamic(%arg0: index) -> memref<?xf32> {
  %0 = memref.alloc(%arg0) : memref<?xf32>
  return %0 : memref<?xf32>
}

func.func @alloc_1d_static() -> memref<42xf32> {
  %0 = memref.alloc() : memref<42xf32>
  return %0 : memref<42xf32>
}

func.func @alloc_3d_dynamic(%arg0: index) -> memref<?x4x42xf32> {
  %0 = memref.alloc(%arg0) : memref<?x4x42xf32>
  return %0 : memref<?x4x42xf32>
}

func.func @alloc_3d_static() -> memref<2x4x42xf32> {
  %0 = memref.alloc() : memref<2x4x42xf32>
  return %0 : memref<2x4x42xf32>
}

func.func @alloca_0d() -> memref<f32> {
  %0 = memref.alloca() : memref<f32>
  return %0 : memref<f32>
}

func.func @alloca_1d_dynamic(%arg0: index) -> memref<?xf32> {
  %0 = memref.alloca(%arg0) : memref<?xf32>
  return %0 : memref<?xf32>
}

func.func @alloca_1d_static() -> memref<42xf32> {
  %0 = memref.alloca() : memref<42xf32>
  return %0 : memref<42xf32>
}

func.func @alloca_3d_dynamic(%arg0: index) -> memref<?x4x42xf32> {
  %0 = memref.alloca(%arg0) : memref<?x4x42xf32>
  return %0 : memref<?x4x42xf32>
}

func.func @alloca_3d_static() -> memref<2x4x42xf32> {
  %0 = memref.alloca() : memref<2x4x42xf32>
  return %0 : memref<2x4x42xf32>
}

func.func @dealloc_0d(%arg0: memref<f32>) {
  memref.dealloc %arg0 : memref<f32>
  return
}

func.func @dealloc_1d_dynamic(%arg0: memref<?xf32>) {
  memref.dealloc %arg0 : memref<?xf32>
  return
}

func.func @dealloc_1d_static(%arg0: memref<42xf32>) {
  memref.dealloc %arg0 : memref<42xf32>
  return
}

func.func @dealloc_3d_dynamic(%arg0: memref<?x4x42xf32>) {
  memref.dealloc %arg0 : memref<?x4x42xf32>
  return
}

func.func @dealloc_3d_static(%arg0: memref<2x4x42xf32>) {
  memref.dealloc %arg0 : memref<2x4x42xf32>
  return
}

func.func @load_0d(%arg0: memref<f32>) -> f32 {
  %0 = memref.load %arg0[] : memref<f32>
  return %0 : f32
}

func.func @load_1d_dynamic(%arg0: memref<?xf32>, %arg1: index) -> f32 {
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  return %0 : f32
}

func.func @load_1d_static(%arg0: memref<42xf32>, %arg1: index) -> f32 {
  %0 = memref.load %arg0[%arg1] : memref<42xf32>
  return %0 : f32
}

func.func @load_3d_dynamic(%arg0: memref<?x4x42xf32>, %arg1: index, %arg2: index, %arg3: index) -> f32 {
  %0 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<?x4x42xf32>
  return %0 : f32
}

func.func @load_3d_static(%arg0: memref<2x4x42xf32>, %arg1: index, %arg2: index, %arg3: index) -> f32 {
  %0 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<2x4x42xf32>
  return %0 : f32
}

func.func @store_0d(%arg0: memref<f32>, %value: f32) {
  memref.store %value,  %arg0[] : memref<f32>
  return
}

func.func @store_1d_dynamic(%arg0: memref<?xf32>, %arg1: index, %value: f32) {
  memref.store %value,  %arg0[%arg1] : memref<?xf32>
  return
}

func.func @store_1d_static(%arg0: memref<42xf32>, %arg1: index, %value: f32) {
  memref.store %value,  %arg0[%arg1] : memref<42xf32>
  return
}

func.func @store_3d_dynamic(%arg0: memref<?x4x42xf32>, %arg1: index, %arg2: index, %arg3: index, %value: f32) {
  memref.store %value,  %arg0[%arg1, %arg2, %arg3] : memref<?x4x42xf32>
  return
}

func.func @store_3d_static(%arg0: memref<2x4x42xf32>, %arg1: index, %arg2: index, %arg3: index, %value: f32) {
  memref.store %value,  %arg0[%arg1, %arg2, %arg3] : memref<2x4x42xf32>
  return
}

// CHECK-LABEL:   llvm.func @free(!llvm.ptr)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.mlir.global external constant @glob_1d(dense<1.010000e+01> : tensor<42xf32>) {addr_space = 0 : i32} : !llvm.array<42 x f32>
// CHECK:         llvm.mlir.global external @glob_2d(dense<4.200000e+00> : tensor<10x5xf32>) {addr_space = 0 : i32} : !llvm.array<10 x array<5 x f32>>

// CHECK-LABEL:   llvm.func @global_1d(
// CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> f32 {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.addressof @glob_1d : !llvm.ptr
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> f32
// CHECK:           llvm.return %[[VAL_4]] : f32
// CHECK:         }

// CHECK-LABEL:   llvm.func @global_2d(
// CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                         %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                         %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.addressof @glob_2d : !llvm.ptr
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_0]], %[[VAL_1]]] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<5 x f32>
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_5]] : f32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloc_0d() -> !llvm.ptr {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_0]], %[[VAL_1]]  : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @malloc(%[[VAL_2]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloc_1d_dynamic(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_0]], %[[VAL_1]]  : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @malloc(%[[VAL_2]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloc_1d_static() -> !llvm.ptr {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(42 : index) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_0]], %[[VAL_1]]  : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @malloc(%[[VAL_2]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloc_3d_dynamic(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(168 : index) : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_0]], %[[VAL_2]]  : i64
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @malloc(%[[VAL_4]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_5]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloc_3d_static() -> !llvm.ptr {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(168 : index) : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_1]], %[[VAL_2]]  : i64
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mul %[[VAL_3]], %[[VAL_0]]  : i64
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @malloc(%[[VAL_4]]) : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_5]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloca_0d() -> !llvm.ptr {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.alloca %[[VAL_0]] x f32 : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloca_1d_dynamic(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.alloca %[[VAL_0]] x f32 : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloca_1d_static() -> !llvm.ptr {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(42 : index) : i64
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.alloca %[[VAL_0]] x f32 : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloca_3d_dynamic(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.alloca %[[VAL_0]] x !llvm.array<4 x array<42 x f32>> : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @alloca_3d_static() -> !llvm.ptr {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.alloca %[[VAL_0]] x !llvm.array<4 x array<42 x f32>> : (i64) -> !llvm.ptr
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @dealloc_0d(
// CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @dealloc_1d_dynamic(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @dealloc_1d_static(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @dealloc_3d_dynamic(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @dealloc_3d_static(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @load_0d(
// CHECK-SAME:                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) -> f32 {
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]][] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> f32
// CHECK:           llvm.return %[[VAL_2]] : f32
// CHECK:         }

// CHECK-LABEL:   llvm.func @load_1d_dynamic(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> f32 {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> f32
// CHECK:           llvm.return %[[VAL_3]] : f32
// CHECK:         }

// CHECK-LABEL:   llvm.func @load_1d_static(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> f32 {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> f32
// CHECK:           llvm.return %[[VAL_3]] : f32
// CHECK:         }

// CHECK-LABEL:   llvm.func @load_3d_dynamic(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                               %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> f32 {
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<42 x f32>>
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> f32
// CHECK:           llvm.return %[[VAL_5]] : f32
// CHECK:         }

// CHECK-LABEL:   llvm.func @load_3d_static(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                              %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                              %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> f32 {
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<42 x f32>>
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> f32
// CHECK:           llvm.return %[[VAL_5]] : f32
// CHECK:         }

// CHECK-LABEL:   llvm.func @store_0d(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                        %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]][] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_2]] : f32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @store_1d_dynamic(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_3]] : f32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @store_1d_static(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_3]] : f32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @store_3d_dynamic(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<42 x f32>>
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_5]] : f32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @store_3d_static(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                               %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                               %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<42 x f32>>
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_5]] : f32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

