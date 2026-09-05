// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite | FileCheck %s

#id = affine_map<(d0) -> (d0)>
#row = affine_map<(d0)[s0] -> (s0, d0)>
#scalar = affine_map<(d0)[s0] -> (s0)>
#coord = affine_map<(d0)[s0, s1] -> (s0, s1)>
#diag = affine_map<(d0)[s0] -> (s0, s0)>

module {
  func.func @shifted_histogram(%samples: memref<64xi32>,
                               %histogram: memref<16xi32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %shift = arith.constant 2 : i32
    %extent = arith.constant 16 : index
    %histogram_view = polygeist.submap(%histogram, %extent) {map = #id}
        : (memref<16xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#id], iterator_types = ["parallel"]}
        outs(%histogram_view : memref<?xi32>) {
    ^bb0(%old: i32):
      linalg.yield %zero : i32
    }
    affine.for %i = 0 to 64 {
      %sample = affine.load %samples[%i] : memref<64xi32>
      %bin32 = arith.shrsi %sample, %shift : i32
      %bin = arith.index_cast %bin32 : i32 to index
      %old = memref.load %histogram[%bin] : memref<16xi32>
      %next = arith.addi %old, %one : i32
      memref.store %next, %histogram[%bin] : memref<16xi32>
    }
    return
  }

  // An additional output makes this a fused scatter, not a pure histogram.
  func.func @histogram_with_side_effect(%samples: memref<64xi32>,
                                        %histogram: memref<16xi32>,
                                        %copy: memref<64xi32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    affine.for %i = 0 to 16 {
      affine.store %zero, %histogram[%i] : memref<16xi32>
    }
    affine.for %i = 0 to 64 {
      %sample = affine.load %samples[%i] : memref<64xi32>
      %bin = arith.index_cast %sample : i32 to index
      %old = memref.load %histogram[%bin] : memref<16xi32>
      %next = arith.addi %old, %one : i32
      memref.store %next, %histogram[%bin] : memref<16xi32>
      affine.store %sample, %copy[%i] : memref<64xi32>
    }
    return
  }

  // Pointer arguments arrive as dynamic memrefs.  The count-loop bound and
  // complete zero-filled submap still prove the two extents needed by CUB.
  func.func @dynamic_histogram(%samples: memref<?xi32>,
                               %histogram: memref<?xi32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %shift = arith.constant 2 : i32
    %extent = arith.constant 16 : index
    %histogram_view = polygeist.submap(%histogram, %extent) {map = #id}
        : (memref<?xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#id], iterator_types = ["parallel"]}
        outs(%histogram_view : memref<?xi32>) {
    ^bb0(%old: i32):
      linalg.yield %zero : i32
    }
    affine.for %i = 0 to 64 {
      %sample = affine.load %samples[%i] : memref<?xi32>
      %bin32 = arith.shrsi %sample, %shift : i32
      %bin = arith.index_cast %bin32 : i32 to index
      %old = memref.load %histogram[%bin] : memref<?xi32>
      %next = arith.addi %old, %one : i32
      memref.store %next, %histogram[%bin] : memref<?xi32>
    }
    return
  }

  func.func @trisolv(%n: i32, %a: memref<?x400xf64>, %x: memref<?xf64>,
                     %b: memref<?xf64>) {
    %one = arith.constant 1 : index
    %bt = bufferization.to_tensor %b restrict : memref<?xf64>
    %ni = arith.index_cast %n : i32 to index
    affine.for %i = 0 to %ni {
      %bv = tensor.extract %bt[%i] : tensor<?xf64>
      affine.store %bv, %x[%i] : memref<?xf64>
      %last = arith.subi %ni, %one : index
      %arow = polygeist.submap(%a, %i, %last) {map = #row} : (memref<?x400xf64>, index, index) -> memref<?xf64>
      %xs = polygeist.submap(%x, %last) {map = #id} : (memref<?xf64>, index) -> memref<?xf64>
      %xo = polygeist.submap(%x, %i, %last) {map = #scalar} : (memref<?xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["reduction"]}
          ins(%arow, %xs : memref<?xf64>, memref<?xf64>)
          outs(%xo : memref<?xf64>) {
      ^bb0(%av: f64, %xv: f64, %acc: f64):
        %product = arith.mulf %av, %xv : f64
        %sub = arith.subf %acc, %product : f64
        %j = linalg.index 0 : index
        %before = arith.cmpi slt, %j, %i : index
        %selected = arith.select %before, %sub, %acc : f64
        linalg.yield %selected : f64
      }
      %sum = affine.load %x[%i] : memref<?xf64>
      %diagonal = affine.load %a[%i, %i] : memref<?x400xf64>
      %result = arith.divf %sum, %diagonal : f64
      affine.store %result, %x[%i] : memref<?xf64>
    }
    return
  }

  func.func @cholesky(%n: i32, %a: memref<?x400xf64>) {
    %one = arith.constant 1 : index
    %ni = arith.index_cast %n : i32 to index
    affine.for %i = 0 to %ni {
      affine.for %j = 0 to #id(%i) {
        %last = arith.subi %i, %one : index
        %left = polygeist.submap(%a, %i, %last) {map = #row} : (memref<?x400xf64>, index, index) -> memref<?xf64>
        %right = polygeist.submap(%a, %j, %last) {map = #row} : (memref<?x400xf64>, index, index) -> memref<?xf64>
        %out = polygeist.submap(%a, %i, %j, %last) {map = #coord} : (memref<?x400xf64>, index, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["reduction"]}
            ins(%left, %right : memref<?xf64>, memref<?xf64>)
            outs(%out : memref<?xf64>) {
        ^bb0(%lv: f64, %rv: f64, %acc: f64):
          %product = arith.mulf %lv, %rv : f64
          %sub = arith.subf %acc, %product : f64
          linalg.yield %sub : f64
        }
        %d = affine.load %a[%j, %j] : memref<?x400xf64>
        %v = affine.load %a[%i, %j] : memref<?x400xf64>
        %q = arith.divf %v, %d : f64
        affine.store %q, %a[%i, %j] : memref<?x400xf64>
      }
      %last = arith.subi %ni, %one : index
      %row_view = polygeist.submap(%a, %i, %last) {map = #row} : (memref<?x400xf64>, index, index) -> memref<?xf64>
      %diagonal_view = polygeist.submap(%a, %i, %last) {map = #diag} : (memref<?x400xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#id, #id], iterator_types = ["reduction"]}
          ins(%row_view : memref<?xf64>) outs(%diagonal_view : memref<?xf64>) {
      ^bb0(%v: f64, %acc: f64):
        %square = arith.mulf %v, %v : f64
        %sub = arith.subf %acc, %square : f64
        linalg.yield %sub : f64
      }
      %diagonal = affine.load %a[%i, %i] : memref<?x400xf64>
      %root = math.sqrt %diagonal : f64
      affine.store %root, %a[%i, %i] : memref<?x400xf64>
    }
    return
  }
}

// CHECK-LABEL: func.func @shifted_histogram
// CHECK: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// CHECK-NOT: affine.for
// CHECK-LABEL: func.func @histogram_with_side_effect
// CHECK: affine.for
// CHECK-NOT: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// CHECK-LABEL: func.func @dynamic_histogram
// CHECK: kernel.launch @cubHistogramEvenI32ShiftZero_memref
// CHECK-NOT: affine.for
// CHECK-LABEL: func.func @trisolv
// CHECK: memref.cast {{.*}} : memref<?x400xf64> to memref<?x?xf64>
// CHECK: kernel.launch @cublasDtrsvLowerRowMajor_memref
// CHECK-NOT: linalg.generic
// CHECK: return
// CHECK-LABEL: func.func @cholesky
// CHECK: memref.cast {{.*}} : memref<?x400xf64> to memref<?x?xf64>
// CHECK: kernel.launch @cusolverDnDpotrfLowerRowMajor_memref
// CHECK-NOT: linalg.generic
// CHECK: return
