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
// CHECK-DAG:     %cst = arith.constant 3.000000e+00 : f64
// CHECK-DAG:     %cst_0 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %2 = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %3 = memref.cast %2 : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %4 = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %5 = memref.cast %4 : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     call @_ZN8MyScalarC1Ed(%5, %cst_0) : (memref<?x!llvm.struct<(struct<(f64)>)>>, f64) -> ()
// CHECK-NEXT:     call @_ZN8MyScalarC1Ed(%3, %cst) : (memref<?x!llvm.struct<(struct<(f64)>)>>, f64) -> ()
// CHECK-NEXT:     %6 = affine.load %2[0] : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     affine.store %6, %0[0] : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %7 = call @_ZN8MyScalaraSEOS_(%5, %1) : (memref<?x!llvm.struct<(struct<(f64)>)>>, memref<?x!llvm.struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:     %8 = "polygeist.memref2pointer"(%4) : (memref<1x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %11 = llvm.load %10 : !llvm.ptr<f64>
// CHECK-NEXT:     call @_Z3used(%11) : (f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN8MyScalarC1Ed(%arg0: memref<?x!llvm.struct<(struct<(f64)>)>>, %arg1: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %arg1, %2 : !llvm.ptr<f64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN8MyScalaraSEOS_(%arg0: memref<?x!llvm.struct<(struct<(f64)>)>>, %arg1: memref<?x!llvm.struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(struct<(f64)>)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:     %2 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:     %4 = call @_ZN1SaSEOS_(%1, %3) : (memref<?x!llvm.struct<(f64)>>, memref<?x!llvm.struct<(f64)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:     return %arg0 : memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK-NEXT:   }
// CHECK-NEXT:     func.func private @_Z3used(f64) attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK:   func.func @_ZN1SaSEOS_(%arg0: memref<?x!llvm.struct<(f64)>>, %arg1: memref<?x!llvm.struct<(f64)>>) -> memref<?x!llvm.struct<(f64)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c8_i64 = arith.constant 8 : i64
// CHECK-DAG:     %false = arith.constant false
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %2 = llvm.bitcast %0 : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     %3 = llvm.bitcast %1 : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     "llvm.intr.memcpy"(%2, %3, %c8_i64, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
// CHECK-NEXT:     return %arg0 : memref<?x!llvm.struct<(f64)>>
// CHECK-NEXT:   }
