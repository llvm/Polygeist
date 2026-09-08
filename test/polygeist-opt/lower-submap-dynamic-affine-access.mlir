// RUN: polygeist-opt %s --convert-linalg-to-loops --lower-polygeist-submap --canonicalize | FileCheck %s

#reverse = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#row = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#id = affine_map<(d0) -> (d0)>

module {
  func.func @reverse_prefix(%base: memref<?xf64>, %n: index,
                            %extent: index, %out: memref<?xf64>) {
    %view = polygeist.submap(%base, %n, %extent) {map = #reverse}
      : (memref<?xf64>, index, index) -> memref<?xf64>
    linalg.generic {
      indexing_maps = [#id, #id],
      iterator_types = ["parallel"]
    } ins(%view : memref<?xf64>) outs(%out : memref<?xf64>) {
    ^bb0(%in: f64, %old: f64):
      linalg.yield %in : f64
    }
    return
  }

  func.func @symbolic_row_output(%base: memref<?x?xf64>, %row: index,
                                 %n: index, %extent: index,
                                 %input: memref<?xf64>) {
    %view = polygeist.submap(%base, %row, %n, %extent) {map = #row}
      : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
    linalg.generic {
      indexing_maps = [#id, #id],
      iterator_types = ["parallel"]
    } ins(%input : memref<?xf64>) outs(%view : memref<?xf64>) {
    ^bb0(%in: f64, %old: f64):
      linalg.yield %in : f64
    }
    return
  }

  func.func @dynamic_view_extent(%base: memref<?xf64>, %n: index,
                                 %extent: index) -> index {
    %c0 = arith.constant 0 : index
    %view = polygeist.submap(%base, %n, %extent) {map = #reverse}
      : (memref<?xf64>, index, index) -> memref<?xf64>
    %dim = memref.dim %view, %c0 : memref<?xf64>
    return %dim : index
  }
}

// CHECK-LABEL: func.func @reverse_prefix
// CHECK-NOT: polygeist.submap
// CHECK: scf.for
// CHECK: affine.apply
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<?xf64>

// CHECK-LABEL: func.func @symbolic_row_output
// CHECK-NOT: polygeist.submap
// CHECK: scf.for
// CHECK: affine.apply
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>

// CHECK-LABEL: func.func @dynamic_view_extent
// CHECK-NOT: polygeist.submap
// CHECK: return %{{.*}} : index
