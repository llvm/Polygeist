// RUN: polygeist-opt %s --linalg-debufferize=use-multi-root=true | FileCheck %s
// A scalar destination aliases itself at every iteration. The vector output
// observes intermediate state, so tensorization must retain one scalar state.
// CHECK-LABEL: func.func @carried
// CHECK: outs({{.*}} : tensor<?xf32>, tensor<f32>)
// CHECK: arith.subf
// CHECK: -> (tensor<?xf32>, tensor<f32>)
#identity = affine_map<(i) -> (i)>
#scalar = affine_map<(i) -> ()>
module {
  func.func @carried(%input: memref<?xf32>, %output: memref<?xf32>, %n: index, %seed: f32) {
    %state = memref.alloca() : memref<f32>
    memref.store %seed, %state[] : memref<f32>
    %a = polygeist.submap(%input, %n) {map = #identity} : (memref<?xf32>, index) -> memref<?xf32>
    %b = polygeist.submap(%output, %n) {map = #identity} : (memref<?xf32>, index) -> memref<?xf32>
    %c = polygeist.submap(%state, %n) {map = #scalar} : (memref<f32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["reduction"]}
      ins(%a : memref<?xf32>) outs(%b, %c : memref<?xf32>, memref<?xf32>) {
      ^bb0(%x: f32, %unused: f32, %old: f32):
        %next = arith.subf %x, %old : f32
        linalg.yield %next, %next : f32, f32
    }
    return
  }
}
