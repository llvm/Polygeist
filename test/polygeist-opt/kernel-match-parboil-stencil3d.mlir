// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s 2>&1 | sed '/^\/\/ CHECK/d' | FileCheck %s

#xm = affine_map<(d0, d1, d2)[s0, s1] -> (((d2 + 1) * s0 + d1 + 1) * s1 + d0)>
#xp = affine_map<(d0, d1, d2)[s0, s1] -> (((d2 + 1) * s0 + d1 + 1) * s1 + d0 + 2)>
#ym = affine_map<(d0, d1, d2)[s0, s1] -> (((d2 + 1) * s0 + d1) * s1 + d0 + 1)>
#yp = affine_map<(d0, d1, d2)[s0, s1] -> (((d2 + 1) * s0 + d1 + 2) * s1 + d0 + 1)>
#zm = affine_map<(d0, d1, d2)[s0, s1] -> ((d2 * s0 + d1 + 1) * s1 + d0 + 1)>
#zp = affine_map<(d0, d1, d2)[s0, s1] -> (((d2 + 2) * s0 + d1 + 1) * s1 + d0 + 1)>
#cc = affine_map<(d0, d1, d2)[s0, s1] -> (((d2 + 1) * s0 + d1 + 1) * s1 + d0 + 1)>
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @stencil(%c0: f32, %c1: f32, %A: tensor<?xf32>,
                     %C: tensor<?xf32>, %ny: index, %nx: index,
                     %ox: index, %oy: index, %oz: index) -> tensor<?xf32> {
    %i0 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #zp} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %i1 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #zm} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %i2 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #yp} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %i3 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #ym} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %i4 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #xp} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %i5 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #xm} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %i6 = polygeist.submap(%A, %ny, %nx, %ox, %oy, %oz) {map = #cc} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %o = polygeist.submap(%C, %ny, %nx, %ox, %oy, %oz) {map = #cc} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %g = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id, #id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%i0, %i1, %i2, %i3, %i4, %i5, %i6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%o : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32,
         %in_3: f32, %in_4: f32, %in_5: f32, %out: f32):
      %s0 = arith.addf %in, %in_0 : f32
      %s1 = arith.addf %s0, %in_1 : f32
      %s2 = arith.addf %s1, %in_2 : f32
      %s3 = arith.addf %s2, %in_3 : f32
      %s4 = arith.addf %s3, %in_4 : f32
      %n = arith.mulf %s4, %c1 : f32
      %cv = arith.mulf %in_5, %c0 : f32
      %v = arith.subf %n, %cv : f32
      linalg.yield %v : f32
    } -> tensor<?x?x?xf32>
    %r = polygeist.submapInverse(%C, %g, %ny, %nx, %ox, %oy, %oz) {map = #cc} : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index, index, index) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}

// CHECK: %r = kernel.launch @cudnnStencil3D7pt_f32_flat_tensor
// CHECK-NOT: submapInverse
