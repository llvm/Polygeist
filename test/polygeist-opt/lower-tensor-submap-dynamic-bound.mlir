// RUN: polygeist-opt %s --lower-polygeist-submap --canonicalize-polygeist | FileCheck %s
#id=affine_map<(i)->(i)>
func.func @narrow_dynamic(%base:tensor<?xf32>,%n:index)->index {
 %c0=arith.constant 0:index
 %view=polygeist.submap(%base,%n){map=#id}:(tensor<?xf32>,index)->tensor<?xf32>
 %size=tensor.dim %view,%c0:tensor<?xf32>
 return %size:index
}
// CHECK-LABEL: func.func @narrow_dynamic
// CHECK-NOT: tensor.dim
// CHECK: return %arg1 : index
func.func @full_dynamic(%base:tensor<?xf32>)->tensor<?xf32> {
 %c0=arith.constant 0:index
 %n=tensor.dim %base,%c0:tensor<?xf32>
 %view=polygeist.submap(%base,%n){map=#id}:(tensor<?xf32>,index)->tensor<?xf32>
 return %view:tensor<?xf32>
}
// CHECK-LABEL: func.func @full_dynamic
// CHECK-NOT: tensor.extract_slice
// CHECK: return %arg0 : tensor<?xf32>
