// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s
#matrix=affine_map<(r,k)->(r,k)>
#row=affine_map<(r,k)->(r)>
#id=affine_map<(r)->(r)>
func.func @row_state(%x:tensor<2x3xf32>,%out:tensor<2x3xf32>,%base:tensor<2xf32>,%seed:f32)->tensor<2x3xf32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %init=linalg.generic {indexing_maps=[#id],iterator_types=["parallel"]} outs(%base:tensor<2xf32>) {
 ^bb0(%old:f32):
  linalg.yield %seed:f32
 } -> tensor<2xf32>
 %state=polygeist.submap(%init,%c2,%c3) {map=#row} : (tensor<2xf32>,index,index)->tensor<2x3xf32>
 %values:2=linalg.generic {indexing_maps=[#matrix,#matrix,#matrix],iterator_types=["parallel","reduction"]} ins(%x:tensor<2x3xf32>) outs(%out,%state:tensor<2x3xf32>,tensor<2x3xf32>) {
 ^bb0(%in:f32,%old:f32,%v:f32):
  %r=arith.mulf %v,%in:f32
  linalg.yield %r,%r:f32,f32
 } -> (tensor<2x3xf32>,tensor<2x3xf32>)
 return %values#0:tensor<2x3xf32>
}
// CHECK-LABEL: func.func @row_state
// CHECK: scf.for {{.*}} iter_args({{.*}}, %[[PRIVATE:.*]] = {{.*}})
// CHECK: linalg.generic
// CHECK: iterator_types = ["reduction"]
// CHECK: tensor<f32>
// CHECK: arith.mulf
// CHECK: linalg.yield
// CHECK: scf.yield {{.*}}, %[[PRIVATE]] :
// CHECK: return

// A result exposed to the caller has no private-state normalization permission.
func.func @observed_state(%x:tensor<2x3xi32>,%out:tensor<2x3xi32>,%base:tensor<2xi32>)->(tensor<2x3xi32>,tensor<2x3xi32>) {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %state=polygeist.submap(%base,%c2,%c3) {map=#row} : (tensor<2xi32>,index,index)->tensor<2x3xi32>
 %values:2=linalg.generic {indexing_maps=[#matrix,#matrix,#matrix],iterator_types=["parallel","reduction"]} ins(%x:tensor<2x3xi32>) outs(%out,%state:tensor<2x3xi32>,tensor<2x3xi32>) {
 ^bb0(%in:i32,%old:i32,%v:i32):
  %r=arith.addi %v,%in:i32
  linalg.yield %r,%r:i32,i32
 } -> (tensor<2x3xi32>,tensor<2x3xi32>)
 return %values#0,%values#1:tensor<2x3xi32>,tensor<2x3xi32>
}
// CHECK-LABEL: func.func @observed_state
// CHECK: polygeist.submap(
// CHECK-NOT: tensor<i32>
// CHECK: return

// Explicit bounded views refine dynamic types on every operand together.
func.func @bounded_state(%x:tensor<?x?xf32>,%out:tensor<?x?xf32>,%base:tensor<2xf32>)->tensor<2x3xf32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %a=polygeist.submap(%x,%c2,%c3) {map=#matrix} : (tensor<?x?xf32>,index,index)->tensor<?x?xf32>
 %b=polygeist.submap(%out,%c2,%c3) {map=#matrix} : (tensor<?x?xf32>,index,index)->tensor<?x?xf32>
 %state=polygeist.submap(%base,%c2,%c3) {map=#row} : (tensor<2xf32>,index,index)->tensor<?x?xf32>
 %values:2=linalg.generic {indexing_maps=[#matrix,#matrix,#matrix],iterator_types=["parallel","reduction"]} ins(%a:tensor<?x?xf32>) outs(%b,%state:tensor<?x?xf32>,tensor<?x?xf32>) {
 ^bb0(%in:f32,%old:f32,%v:f32):
  %r=arith.addf %v,%in:f32
  linalg.yield %r,%r:f32,f32
 } -> (tensor<?x?xf32>,tensor<?x?xf32>)
 %result=tensor.cast %values#0:tensor<?x?xf32> to tensor<2x3xf32>
 return %result:tensor<2x3xf32>
}
// CHECK-LABEL: func.func @bounded_state
// CHECK: scf.for
// CHECK: linalg.generic
// CHECK: iterator_types = ["reduction"]
// CHECK: tensor<3xf32>, tensor<f32>
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK: return

// A broadcast parallel coordinate is shared state, not an independent batch.
func.func @shared_parallel_state(%x:tensor<2x3xi32>,%out:tensor<2x3xi32>,%base:tensor<1xi32>)->tensor<2x3xi32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %state=polygeist.submap(%base,%c2,%c3) {map=affine_map<(r,k)->(0)>} : (tensor<1xi32>,index,index)->tensor<2x3xi32>
 %values:2=linalg.generic {indexing_maps=[#matrix,#matrix,#matrix],iterator_types=["parallel","reduction"]} ins(%x:tensor<2x3xi32>) outs(%out,%state:tensor<2x3xi32>,tensor<2x3xi32>) {
 ^bb0(%in:i32,%old:i32,%v:i32):
  %r=arith.addi %v,%in:i32
  linalg.yield %r,%r:i32,i32
 } -> (tensor<2x3xi32>,tensor<2x3xi32>)
 return %values#0:tensor<2x3xi32>
}
// CHECK-LABEL: func.func @shared_parallel_state
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK-NOT: scf.for
// CHECK: return
