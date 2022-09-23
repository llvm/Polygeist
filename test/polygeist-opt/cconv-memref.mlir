// RUN: polygeist-opt -convert-polygeist-to-llvm %s | FileCheck %s

// CHECK: llvm.mlir.global external constant @glob_1d({{.*}})
// CHECK-SAME: !llvm.array<42 x f32>
memref.global constant @glob_1d : memref<42xf32> = dense<10.1>

// CHECK: llvm.mlir.global external @glob_2d({{.*}})
// CHECK-SAME: !llvm.array<10 x array<5 x f32>>
memref.global @glob_2d : memref<10x5xf32> = dense<4.2>

// CHECK-LABEL: @global_1d
// CHECK: %[[whole_address:.+]] = llvm.mlir.addressof @glob_1d : !llvm.ptr<array<42 x f32>>
// CHECK: %[[address:.+]] = llvm.getelementptr %[[whole_address]][0, 0] : {{.*}} -> !llvm.ptr<f32>
// CHECK: llvm.getelementptr %[[address]][%{{.*}}]
func.func @global_1d(%arg1: index) -> f32 {
  %1 = memref.get_global @glob_1d : memref<42xf32>
  %2 = memref.load %1[%arg1] : memref<42xf32>
  return %2 : f32
}

// CHECK-LABEL: @global_2d
// CHECK: %[[whole_address:.+]] = llvm.mlir.addressof @glob_2d : !llvm.ptr<array<10 x array<5 x f32>>>
// CHECK: %[[address:.+]] = llvm.getelementptr %[[whole_address]][0, 0] : {{.*}} -> !llvm.ptr<array<5 x f32>>
// CHECK: llvm.getelementptr %[[address]][%{{.*}}, %{{.*}}]
func.func @global_2d(%arg0: index, %arg1: index, %value: f32) {
  %1 = memref.get_global @glob_2d : memref<10x5xf32>
  memref.store %value, %1[%arg0, %arg1] : memref<10x5xf32>
  return
}

// CHECK-LABEL: @alloc_0d
// CHECK: %[[num_elems:.+]] = llvm.mlir.constant(1 : index)
// CHECK: %[[null:.+]] = llvm.mlir.null
// CHECK: %[[offset_one:.+]] = llvm.getelementptr %[[null]][1]
// CHECK: %[[elem_size:.+]] = llvm.ptrtoint %[[offset_one]]
// CHECK: %[[byte_size:.+]] = llvm.mul %[[num_elems]], %[[elem_size]]
// CHECK: llvm.call @malloc(%[[byte_size]])
func.func @alloc_0d() -> memref<f32> {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// CHECK-LABEL: @alloc_1d_dynamic
// CHECK-SAME: %[[num_elems:.+]]: i{{.*}}
// CHECK: %[[null:.+]] = llvm.mlir.null
// CHECK: %[[offset_one:.+]] = llvm.getelementptr %[[null]][1]
// CHECK: %[[elem_size:.+]] = llvm.ptrtoint %[[offset_one]]
// CHECK: %[[byte_size:.+]] = llvm.mul %[[num_elems]], %[[elem_size]]
// CHECK: llvm.call @malloc(%[[byte_size]])
func.func @alloc_1d_dynamic(%arg0: index) -> memref<?xf32> {
  %0 = memref.alloc(%arg0) : memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: @alloc_1d_static
// CHECK: %[[num_elems:.+]] = llvm.mlir.constant(42 : index)
// CHECK: %[[null:.+]] = llvm.mlir.null
// CHECK: %[[offset_one:.+]] = llvm.getelementptr %[[null]][1]
// CHECK: %[[elem_size:.+]] = llvm.ptrtoint %[[offset_one]]
// CHECK: %[[byte_size:.+]] = llvm.mul %[[num_elems]], %[[elem_size]]
// CHECK: llvm.call @malloc(%[[byte_size]])
func.func @alloc_1d_static() -> memref<42xf32> {
  %0 = memref.alloc() : memref<42xf32>
  return %0 : memref<42xf32>
}

// CHECK-LABEL: @alloc_3d_dynamic
// CHECK-SAME: %[[num_outer_elems:.+]]: i{{.*}}
// CHECK: %[[num_static_elems:.+]] = llvm.mlir.constant(168 : index)
// CHECK: %[[num_elems:.+]] = llvm.mul %[[num_outer_elems]], %[[num_static_elems]]
// CHECK: %[[null:.+]] = llvm.mlir.null
// CHECK: %[[offset_one:.+]] = llvm.getelementptr %[[null]][1]
// CHECK: %[[elem_size:.+]] = llvm.ptrtoint %[[offset_one]]
// CHECK: %[[byte_size:.+]] = llvm.mul %[[num_elems]], %[[elem_size]]
// CHECK: llvm.call @malloc(%[[byte_size]])
func.func @alloc_3d_dynamic(%arg0: index) -> memref<?x4x42xf32> {
  %0 = memref.alloc(%arg0) : memref<?x4x42xf32>
  return %0 : memref<?x4x42xf32>
}

// CHECK-LABEL: @alloc_3d_static
// CHECK: %[[num_outer_elems:.+]] = llvm.mlir.constant(2 : index)
// CHECK: %[[num_static_elems:.+]] = llvm.mlir.constant(168 : index)
// CHECK: %[[num_elems:.+]] = llvm.mul %[[num_outer_elems]], %[[num_static_elems]]
// CHECK: %[[null:.+]] = llvm.mlir.null
// CHECK: %[[offset_one:.+]] = llvm.getelementptr %[[null]][1]
// CHECK: %[[elem_size:.+]] = llvm.ptrtoint %[[offset_one]]
// CHECK: %[[byte_size:.+]] = llvm.mul %[[num_elems]], %[[elem_size]]
// CHECK: llvm.call @malloc(%[[byte_size]])
func.func @alloc_3d_static() -> memref<2x4x42xf32> {
  %0 = memref.alloc() : memref<2x4x42xf32>
  return %0 : memref<2x4x42xf32>
}

// CHECK-LABEL: @alloca_0d
// CHECK: %[[num_elems:.+]] = llvm.mlir.constant(1 : index)
// CHECK: llvm.alloca %[[num_elems]] x f32
func.func @alloca_0d() {
  %0 = memref.alloca() : memref<f32>
  return
}

// CHECK-LABEL: @alloca_1d_dynamic
// CHECK-SAME: %[[num_elems:.+]]: i{{.*}}
// CHECK: llvm.alloca %[[num_elems]] x f32
func.func @alloca_1d_dynamic(%arg0: index) {    
  %0 = memref.alloca(%arg0) : memref<?xf32>
  return
}

// CHECK-LABEL: @alloca_1d_static
// CHECK: %[[num_elems:.+]] = llvm.mlir.constant(42 : index)
// CHECK: llvm.alloca %[[num_elems]] x f32
func.func @alloca_1d_static() {
  %0 = memref.alloca() : memref<42xf32>
  return
}

// CHECK-LABEL: @alloca_3d_dynamic
// CHECK-SAME: %[[num_elems:.+]]: i{{.*}}
// CHECK: llvm.alloca %[[num_elems]] x !llvm.array<4 x array<42 x f32>>
func.func @alloca_3d_dynamic(%arg0: index) {
  %0 = memref.alloca(%arg0) : memref<?x4x42xf32>
  return
}

// CHECK-LABEL: @alloca_3d_static
// CHECK: %[[num_elems:.+]] = llvm.mlir.constant(2 : index)
// CHECK: llvm.alloca %[[num_elems]] x !llvm.array<4 x array<42 x f32>>
func.func @alloca_3d_static() {
  %0 = memref.alloca() : memref<2x4x42xf32>
  return
}

// CHECK-LABEL: @dealloc_0d
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr
// CHECK: %[[casted:.+]] = llvm.bitcast %[[memref]]
// CHECK: llvm.call @free(%[[casted]])
func.func @dealloc_0d(%arg0: memref<f32>) {
  memref.dealloc %arg0 : memref<f32>
  return
}

// CHECK-LABEL: @dealloc_1d_dynamic
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr
// CHECK: %[[casted:.+]] = llvm.bitcast %[[memref]]
// CHECK: llvm.call @free(%[[casted]])
func.func @dealloc_1d_dynamic(%arg0: memref<?xf32>) {
  memref.dealloc %arg0 : memref<?xf32>
  return
}

// CHECK-LABEL: @dealloc_1d_static
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr
// CHECK: %[[casted:.+]] = llvm.bitcast %[[memref]]
// CHECK: llvm.call @free(%[[casted]])
func.func @dealloc_1d_static(%arg0: memref<42xf32>) {
  memref.dealloc %arg0 : memref<42xf32>
  return
}

// CHECK-LABEL: @dealloc_3d_dynamic
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr
// CHECK: %[[casted:.+]] = llvm.bitcast %[[memref]]
// CHECK: llvm.call @free(%[[casted]])
func.func @dealloc_3d_dynamic(%arg0: memref<?x4x42xf32>) {
  memref.dealloc %arg0 : memref<?x4x42xf32>
  return
}

// CHECK-LABEL: @dealloc_3d_static
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr
// CHECK: %[[casted:.+]] = llvm.bitcast %[[memref]]
// CHECK: llvm.call @free(%[[casted]])
func.func @dealloc_3d_static(%arg0: memref<2x4x42xf32>) {
  memref.dealloc %arg0 : memref<2x4x42xf32>
  return
}

// CHECK-LABEL: @load_0d
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][]
// CHECK: llvm.load %[[address]]
func.func @load_0d(%arg0: memref<f32>) -> f32 {
  %0 = memref.load %arg0[] : memref<f32>
  return %0 : f32
}

// CHECK-LABEL: @load_1d_dynamic
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx:.+]]: i{{.*}}
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx]]]
// CHECK: llvm.load %[[address]]
func.func @load_1d_dynamic(%arg0: memref<?xf32>, %arg1: index) -> f32 {
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  return %0 : f32
}

// CHECK-LABEL: @load_1d_static
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx:.+]]: i{{.*}}
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx]]]
// CHECK: llvm.load %[[address]]
func.func @load_1d_static(%arg0: memref<42xf32>, %arg1: index) -> f32 {
  %0 = memref.load %arg0[%arg1] : memref<42xf32>
  return %0 : f32
}

// CHECK-LABEL: @load_3d_dynamic
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx1:.+]]: i{{.*}}, %[[idx2:.+]]: i{{.*}}, %[[idx3:.+]]: i{{.*}}
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx1]], %[[idx2]], %[[idx3]]
// CHECK: llvm.load %[[address]]
func.func @load_3d_dynamic(%arg0: memref<?x4x42xf32>, %arg1: index, %arg2: index, %arg3: index) -> f32 {
  %0 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<?x4x42xf32>
  return %0 : f32
}

// CHECK-LABEL: @load_3d_static
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx1:.+]]: i{{.*}}, %[[idx2:.+]]: i{{.*}}, %[[idx3:.+]]: i{{.*}}
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx1]], %[[idx2]], %[[idx3]]
// CHECK: llvm.load %[[address]]
func.func @load_3d_static(%arg0: memref<2x4x42xf32>, %arg1: index, %arg2: index, %arg3: index) -> f32 {
  %0 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<2x4x42xf32>
  return %0 : f32
}

// CHECK-LABEL: @store_0d
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[value:.+]]: f32
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][]
// CHECK: llvm.store %[[value]], %[[address]]
func.func @store_0d(%arg0: memref<f32>, %value: f32) {
  memref.store %value,  %arg0[] : memref<f32>
  return
}

// CHECK-LABEL: @store_1d_dynamic
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx:.+]]: i{{.*}}, %[[value:.+]]: f32
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx]]]
// CHECK: llvm.store %[[value]], %[[address]]
func.func @store_1d_dynamic(%arg0: memref<?xf32>, %arg1: index, %value: f32) {
  memref.store %value,  %arg0[%arg1] : memref<?xf32>
  return
}

// CHECK-LABEL: @store_1d_static
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx:.+]]: i{{.*}}, %[[value:.+]]: f32
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx]]]
// CHECK: llvm.store %[[value]], %[[address]]
func.func @store_1d_static(%arg0: memref<42xf32>, %arg1: index, %value: f32) {
  memref.store %value,  %arg0[%arg1] : memref<42xf32>
  return
}

// CHECK-LABEL: @store_3d_dynamic
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx1:.+]]: i{{.*}}, %[[idx2:.+]]: i{{.*}}, %[[idx3:.+]]: i{{.*}}, %[[value:.+]]: f32
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx1]], %[[idx2]], %[[idx3]]
// CHECK: llvm.store %[[value]], %[[address]]
func.func @store_3d_dynamic(%arg0: memref<?x4x42xf32>, %arg1: index, %arg2: index, %arg3: index, %value: f32) {
  memref.store %value,  %arg0[%arg1, %arg2, %arg3] : memref<?x4x42xf32>
  return
}

// CHECK-LABEL: @store_3d_static
// CHECK-SAME: %[[memref:.+]]: !llvm.ptr{{.*}}, %[[idx1:.+]]: i{{.*}}, %[[idx2:.+]]: i{{.*}}, %[[idx3:.+]]: i{{.*}}, %[[value:.+]]: f32
// CHECK: %[[address:.+]] = llvm.getelementptr %[[memref]][%[[idx1]], %[[idx2]], %[[idx3]]
// CHECK: llvm.store %[[value]], %[[address]]
func.func @store_3d_static(%arg0: memref<2x4x42xf32>, %arg1: index, %arg2: index, %arg3: index, %value: f32) {
  memref.store %value,  %arg0[%arg1, %arg2, %arg3] : memref<2x4x42xf32>
  return
}
