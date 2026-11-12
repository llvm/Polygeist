// RUN: polygeist-opt %s --linalg-debufferize="use-multi-root=true" --canonicalize | FileCheck %s
#row=affine_map<(k)[r]->(r,k)>
#scalar=affine_map<(k)->()>
#id=affine_map<(k)->(k)>
func.func @local_multi_output(%x:memref<2x3xi32>,%y:memref<2x3xi32>,%totals:memref<2xi32>) {
 %c3=arith.constant 3:index
 %seed=arith.constant 17:i32
 affine.for %r=0 to 2 {
  %state=memref.alloca():memref<i32>
  affine.store %seed,%state[]:memref<i32>
  %input=polygeist.submap(%x,%r,%c3) {map=#row}:(memref<2x3xi32>,index,index)->memref<3xi32>
  %output=polygeist.submap(%y,%r,%c3) {map=#row}:(memref<2x3xi32>,index,index)->memref<3xi32>
  %broadcast=polygeist.submap(%state,%c3) {map=#scalar}:(memref<i32>,index)->memref<3xi32>
  linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["reduction"]} ins(%input:memref<3xi32>) outs(%output,%broadcast:memref<3xi32>,memref<3xi32>) {
  ^bb0(%in:i32,%old:i32,%acc:i32):
   %next=arith.addi %acc,%in:i32
   linalg.yield %next,%next:i32,i32
  }
  %sum=affine.load %state[]:memref<i32>
  affine.store %sum,%totals[%r]:memref<2xi32>
 }
 return
}
// CHECK-LABEL: func.func @local_multi_output
// CHECK: affine.for
// CHECK: tensor.insert
// CHECK: linalg.generic
// CHECK: outs({{.*}}tensor<3xi32>, tensor<i32>)
// CHECK: linalg.yield
// CHECK: tensor.extract
// CHECK: affine.yield
// CHECK: return

// Scratch created in an outer iteration may be loop carried by an inner loop.
func.func @nested_private(%out:memref<2xi32>) {
 %c0=arith.constant 0:index
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %c1=arith.constant 1:index
 %seed=arith.constant 19:i32
 %one=arith.constant 1:i32
 scf.for %r=%c0 to %c2 step %c1 {
  %state=memref.alloca():memref<i32>
  memref.store %seed,%state[]:memref<i32>
  scf.for %k=%c0 to %c3 step %c1 {
   %v=memref.load %state[]:memref<i32>
   %n=arith.addi %v,%one:i32
   memref.store %n,%state[]:memref<i32>
  }
  %v=memref.load %state[]:memref<i32>
  memref.store %v,%out[%r]:memref<2xi32>
 }
 return
}
// CHECK-LABEL: func.func @nested_private
// CHECK: scf.for
// CHECK: scf.for {{.*}}iter_args
// CHECK: tensor.extract
// CHECK: arith.addi
// CHECK: tensor.insert
// CHECK: scf.yield
// CHECK: return

// A copy is an external memory observer; do not drop its required writeback.
func.func @observed_copy(%out:memref<i32>) {
 %seed=arith.constant 11:i32
 affine.for %r=0 to 2 {
  %state=memref.alloca():memref<i32>
  affine.store %seed,%state[]:memref<i32>
  memref.copy %state,%out:memref<i32> to memref<i32>
 }
 return
}
// CHECK-LABEL: func.func @observed_copy
// CHECK-NOT: bufferization.to_tensor
// CHECK: affine.store
// CHECK: memref.copy
// CHECK: return
