// RUN: polygeist-opt %s --normalize-linalg-matching-views="split-reduction-inputs=true" | FileCheck %s
#row = affine_map<(r,k)->(r,k)>
#out = affine_map<(r,k)->(r)>
func.func @centered_square(%x:tensor<?x?xf32>,%seed:tensor<?xf32>,%shift:f32) -> tensor<?xf32> {
 %result=linalg.generic {indexing_maps=[#row,#out],iterator_types=["parallel","reduction"]} ins(%x:tensor<?x?xf32>) outs(%seed:tensor<?xf32>) {
 ^bb0(%a:f32,%acc:f32):
  %d=arith.subf %a,%shift:f32
  %s=arith.mulf %d,%d:f32
  %next=arith.addf %acc,%s:f32
  linalg.yield %next:f32
 } -> tensor<?xf32>
 return %result:tensor<?xf32>
}
// CHECK-LABEL: func.func @centered_square
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.subf
// CHECK-NEXT: linalg.yield
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.mulf
// CHECK-NEXT: linalg.yield
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK: return
func.func @state_dependent(%x:tensor<?x?xf32>,%seed:tensor<?xf32>,%scale:f32) -> tensor<?xf32> {
 %result=linalg.generic {indexing_maps=[#row,#out],iterator_types=["parallel","reduction"]} ins(%x:tensor<?x?xf32>) outs(%seed:tensor<?xf32>) {
 ^bb0(%a:f32,%acc:f32):
  %sum=arith.addf %acc,%a:f32
  %next=arith.mulf %sum,%scale:f32
  linalg.yield %next:f32
 } -> tensor<?xf32>
 return %result:tensor<?xf32>
}
// CHECK-LABEL: func.func @state_dependent
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK-NOT: linalg.generic
// CHECK: return
func.func private @external_effect(f32)->f32
func.func @effectful(%x:tensor<?x?xf32>,%seed:tensor<?xf32>) -> tensor<?xf32> {
 %result=linalg.generic {indexing_maps=[#row,#out],iterator_types=["parallel","reduction"]} ins(%x:tensor<?x?xf32>) outs(%seed:tensor<?xf32>) {
 ^bb0(%a:f32,%acc:f32):
  %call=func.call @external_effect(%a):(f32)->f32
  %next=arith.addf %acc,%call:f32
  linalg.yield %next:f32
 } -> tensor<?xf32>
 return %result:tensor<?xf32>
}
// CHECK-LABEL: func.func @effectful
// CHECK: linalg.generic
// CHECK: func.call @external_effect
// CHECK: arith.addf
// CHECK-NOT: linalg.generic
// CHECK: return
