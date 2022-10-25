// RUN: cgeist %s --function=* -S | FileCheck %s

union S {
	double d;
};

class MyScalar {
 public:
  S v;
  MyScalar(double vv) {
   v.d = vv;
  }
};

void use(double);
void meta() {
	MyScalar alpha_scalar(1.0);
	alpha_scalar = MyScalar(3.0);
	use(alpha_scalar.v.d);
}

// CHECK:   func.func @_Z4metav() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 3.000000e+00 : f64
// CHECK-DAG:     %[[cst_0:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V2:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V3:.+]] = memref.cast %[[V2]] : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V4:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V5:.+]] = memref.cast %[[V4]] : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     call @_ZN8MyScalarC1Ed(%[[V5]], %[[cst_0]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>, f64) -> ()
// CHECK-NEXT:     call @_ZN8MyScalarC1Ed(%[[V3]], %[[cst]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>, f64) -> ()
// CHECK-NEXT:     %[[V6:.+]] = affine.load %[[V2]][0] : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     affine.store %[[V6]], %[[V0]][0] : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V7:.+]] = call @_ZN8MyScalaraSEOS_(%[[V5]], %[[V1]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>, memref<?x!llvm.struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V8:.+]] = "polygeist.memref2pointer"(%[[V4]]) : (memref<1x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V9:.+]] = llvm.getelementptr %[[V8]][0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %[[V10:.+]] = llvm.getelementptr %[[V9]][0, 0] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %[[V11:.+]] = llvm.load %[[V10]] : !llvm.ptr<f64>
// CHECK-NEXT:     call @_Z3used(%[[V11]]) : (f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN8MyScalarC1Ed(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(f64)>)>>, %[[arg1:.+]]: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %[[arg1]], %[[V2]] : !llvm.ptr<f64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN8MyScalaraSEOS_(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(f64)>)>>, %[[arg1:.+]]: memref<?x!llvm.struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(struct<(f64)>)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:     %[[V2:.+]] = "polygeist.memref2pointer"(%[[arg1]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %[[V3:.+]] = "polygeist.pointer2memref"(%[[V2]]) : (!llvm.ptr<struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:     %[[V4:.+]] = call @_ZN1SaSEOS_(%[[V1]], %[[V3]]) : (memref<?x!llvm.struct<(f64)>>, memref<?x!llvm.struct<(f64)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:     return %[[arg0]] : memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:   }
// CHECK-NEXT:     func.func private @_Z3used(f64) attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK:   func.func @_ZN1SaSEOS_(%[[arg0:.+]]: memref<?x!llvm.struct<(f64)>>, %[[arg1:.+]]: memref<?x!llvm.struct<(f64)>>) -> memref<?x!llvm.struct<(f64)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %[[c8_i64:.+]] = arith.constant 8 : i64
// CHECK-DAG:     %[[false:.+]] = arith.constant false
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[arg1]]) : (memref<?x!llvm.struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.bitcast %[[V0]] : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     %[[V3:.+]] = llvm.bitcast %[[V1]] : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     "llvm.intr.memcpy"(%[[V2]], %[[V3]], %[[c8_i64]], %[[false]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
// CHECK-NEXT:     return %[[arg0]] : memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:   }
