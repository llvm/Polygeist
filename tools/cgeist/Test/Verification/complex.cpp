// RUN: cgeist %s --function='*' -S | FileCheck %s

int foo() {
    __complex__ float a = 1.0i + 30.0f;
    a = a + a;
    float b = __real__ a;
    float c = __imag__ a;
    return (int) b + (int) c;
    // CHECK: foo
    // CHECK-NEXT: %[[c:.*]] = arith.constant 62
    // CHECK-NEXT: return %[[c]]
}

int bar() {
    __complex__ float a = 1.0i + 30.0f;
    a += a;
    float r = __real__ a + __imag__ a;
    return r;
    // CHECK: bar
    // CHECK-NEXT: %[[c:.*]] = arith.constant 62
    // CHECK-NEXT: return %[[c]]
}

class mcomplex
{
    public:
        mcomplex(double __r, double __i)
        : _M_value{ __r, __i } {}
    private:
        __complex__ double _M_value;
};

mcomplex *baz() {
    // CHECK:  func.func @_Z3bazv() -> memref<?x!llvm.struct<(struct<(f64, f64)>)>>
    // CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f64
    // CHECK-NEXT:    %cst_0 = arith.constant 3.000000e+01 : f64
    // CHECK-NEXT:    %alloc = memref.alloc() : memref<1x!llvm.struct<(struct<(f64, f64)>)>>
    // CHECK-NEXT:    %cast = memref.cast %alloc : memref<1x!llvm.struct<(struct<(f64, f64)>)>> to memref<?x!llvm.struct<(struct<(f64, f64)>)>>
    // CHECK-NEXT:    call @_ZN8mcomplexC1Edd(%cast, %cst, %cst_0) : (memref<?x!llvm.struct<(struct<(f64, f64)>)>>, f64, f64) -> ()
    // CHECK-NEXT:    return %cast : memref<?x!llvm.struct<(struct<(f64, f64)>)>>
    mcomplex *a = new mcomplex(1, 30);
    return a;
}
    // CHECK:  func.func @_ZN8mcomplexC1Edd(%arg0: memref<?x!llvm.struct<(struct<(f64, f64)>)>>, %arg1: f64, %arg2: f64)
    // CHECK-NEXT:    %alloca = memref.alloca() : memref<memref<2xf64>>
    // CHECK-NEXT:    %0 = affine.load %alloca[] : memref<memref<2xf64>>
    // CHECK-NEXT:    affine.store %arg1, %0[0] : memref<2xf64>
    // CHECK-NEXT:    affine.store %arg2, %0[1] : memref<2xf64>
    // CHECK-NEXT:    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(f64, f64)>)>>) -> !llvm.ptr<struct<(struct<(f64, f64)>)>>
    // CHECK-NEXT:    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(struct<(f64, f64)>)>>) -> !llvm.ptr<struct<(f64, f64)>>
    // CHECK-NEXT:    %3 = llvm.bitcast %2 : !llvm.ptr<struct<(f64, f64)>> to !llvm.ptr<memref<2xf64>>
    // CHECK-NEXT:    llvm.store %0, %3 : !llvm.ptr<memref<2xf64>>
    // CHECK-NEXT:    return
