// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=copy_and_launch' %s | FileCheck %s --check-prefix=PREPARE
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=injective_writeback' %s | FileCheck %s --check-prefix=WRITEBACK
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=bufferized_injective_writeback' %s | FileCheck %s --check-prefix=BUFFERIZED-WRITEBACK
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=redundant_roundtrip' %s | FileCheck %s --check-prefix=ROUNDTRIP

module attributes {gpu.container_module} {
  memref.global "private" @workspace : memref<16xf64> {alignment = 4096 : i64}

  gpu.module @kernels {
    gpu.func @kernel(%arg0: memref<16xf64>) kernel {
      gpu.return
    }
  }

  func.func @copy_and_launch(%src: memref<16xf64>,
                             %dst: memref<16xf64>) {
    %c1 = arith.constant 1 : index
    memref.copy %src, %dst : memref<16xf64> to memref<16xf64>
    gpu.launch_func @kernels::@kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%dst : memref<16xf64>)
    return
  }

  func.func @injective_writeback(%base: memref<16xf64>,
                                  %view: memref<4xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %snapshot = memref.alloc() : memref<16xf64>
    memref.copy %base, %snapshot : memref<16xf64> to memref<16xf64>
    %updated = scf.for %i = %c0 to %c4 step %c1
        iter_args(%dst = %snapshot) -> (memref<16xf64>) {
      %value = memref.load %view[%i] : memref<4xf64>
      memref.store %value, %dst[%i] : memref<16xf64>
      scf.yield %dst : memref<16xf64>
    } {polygeist.injective_writeback}
    memref.copy %updated, %base : memref<16xf64> to memref<16xf64>
    return
  }

  func.func @bufferized_injective_writeback(%base: memref<16xf64>,
                                             %view: memref<4xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %snapshot = memref.alloc() : memref<16xf64>
    memref.copy %base, %snapshot : memref<16xf64> to memref<16xf64>
    scf.for %i = %c0 to %c4 step %c1 {
      %value = memref.load %view[%i] : memref<4xf64>
      memref.store %value, %snapshot[%i] : memref<16xf64>
    } {polygeist.injective_writeback}
    memref.copy %snapshot, %base : memref<16xf64> to memref<16xf64>
    return
  }

  func.func @redundant_roundtrip(%base: memref<?xf64>) {
    %cast = memref.cast %base
        : memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
    %c0 = arith.constant 0 : index
    %size = memref.dim %base, %c0 : memref<?xf64>
    %snapshot = memref.alloc(%size) : memref<?xf64>
    memref.copy %cast, %snapshot
        : memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
    memref.copy %snapshot, %base : memref<?xf64> to memref<?xf64>
    return
  }
}

// PREPARE-LABEL: func.func @copy_and_launch
// PREPARE: gpu.host_register
// PREPARE: linalg.copy ins(%{{.*}} : memref<16xf64>) outs(%{{.*}} : memref<16xf64>)
// PREPARE: gpu.launch_func @kernels::@kernel
// PREPARE-SAME: polygeist.cuda_graph_safe

// WRITEBACK-LABEL: func.func @injective_writeback
// WRITEBACK-NOT: memref.alloc
// WRITEBACK-NOT: memref.copy
// WRITEBACK: scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// WRITEBACK: %[[VALUE:.*]] = memref.load %{{.*}}[%[[I]]] : memref<4xf64>
// WRITEBACK: memref.store %[[VALUE]], %{{.*}}[%[[I]]] : memref<16xf64>
// WRITEBACK: return

// BUFFERIZED-WRITEBACK-LABEL: func.func @bufferized_injective_writeback
// BUFFERIZED-WRITEBACK-NOT: memref.alloc
// BUFFERIZED-WRITEBACK-NOT: memref.copy
// BUFFERIZED-WRITEBACK: scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// BUFFERIZED-WRITEBACK: %[[VALUE:.*]] = memref.load %{{.*}}[%[[I]]] : memref<4xf64>
// BUFFERIZED-WRITEBACK: memref.store %[[VALUE]], %{{.*}}[%[[I]]] : memref<16xf64>
// BUFFERIZED-WRITEBACK: return

// ROUNDTRIP-LABEL: func.func @redundant_roundtrip
// ROUNDTRIP-NOT: memref.alloc
// ROUNDTRIP-NOT: memref.copy
// ROUNDTRIP-NOT: linalg.copy
// ROUNDTRIP: return
