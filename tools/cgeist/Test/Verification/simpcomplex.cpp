// RUN: cgeist %s --struct-abi=0 --function='*' -S | FileCheck %s --check-prefix=STRUCT
// COM: we dont support this yet: cgeist %s --function='*' -S | FileCheck %s

void foo() {
    __complex__ float a;
}
__complex__ float *bar() {
    auto a = new __complex__ float;
    return a;
}
float access_imag(__complex__ float a) {
    return __imag__ a;
}
float access_real(__complex__ float a) {
    return __real__ a;
}
float ref_imag(__complex__ float a) {
    __imag__ a = 2.0f;
    return __imag__ a;
}
float ref_real(__complex__ float a) {
    __real__ a = 3.0f;
    return __real__ a;
}
double cast(__complex__ float a) {
    __complex__ double b = a;
    return __real__ b + __imag__ b;
}

float imag_literal() {
    __complex__ float b = 10.0f + 3.0fi;
    return __imag__ b + __real__ b;
}
float imag_literal2() {
    __complex__ float b = 3.0fi;
    return __imag__ b + __real__ b;
}
float add() {
    __complex__ float a = 10.0f + 5.0fi;
    __complex__ float b = 30.0f + 2.0fi;
    __complex__ float c = a + b;
    return __imag__ c + __real__ c;
}
float addassign() {
    __complex__ float a = 10.0f + 5.0fi;
    __complex__ float c = 30.0f + 2.0fi;
    c += a;
    return __imag__ c + __real__ c;
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
    mcomplex *a = new mcomplex(1, 30);
    return a;
}

// STRUCT-LABEL:   func.func @_Z3barv() -> memref<?x!llvm.struct<(f32, f32)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = memref.alloc() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_1:.*]] = memref.cast %[[VAL_0]] : memref<1x!llvm.struct<(f32, f32)>> to memref<?x!llvm.struct<(f32, f32)>>
// STRUCT:           return %[[VAL_1]] : memref<?x!llvm.struct<(f32, f32)>>

// STRUCT-LABEL:   func.func @_Z11access_imagCf(
// STRUCT-SAME:                                 %[[VAL_0:.*]]: !llvm.struct<(f32, f32)>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_1:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           affine.store %[[VAL_0]], %[[VAL_1]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_2:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<f32>
// STRUCT:           return %[[VAL_4]] : f32

// STRUCT-LABEL:   func.func @_Z11access_realCf(
// STRUCT-SAME:                                 %[[VAL_0:.*]]: !llvm.struct<(f32, f32)>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_1:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           affine.store %[[VAL_0]], %[[VAL_1]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_2:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]][0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<f32>
// STRUCT:           return %[[VAL_4]] : f32

// STRUCT-LABEL:   func.func @_Z8ref_imagCf(
// STRUCT-SAME:                             %[[VAL_0:.*]]: !llvm.struct<(f32, f32)>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
// STRUCT:           %[[VAL_2:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           affine.store %[[VAL_0]], %[[VAL_2]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           llvm.store %[[VAL_1]], %[[VAL_4]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<f32>
// STRUCT:           return %[[VAL_5]] : f32

// STRUCT-LABEL:   func.func @_Z8ref_realCf(
// STRUCT-SAME:                             %[[VAL_0:.*]]: !llvm.struct<(f32, f32)>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_1:.*]] = arith.constant 3.000000e+00 : f32
// STRUCT:           %[[VAL_2:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           affine.store %[[VAL_0]], %[[VAL_2]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_3]][0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           llvm.store %[[VAL_1]], %[[VAL_4]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<f32>
// STRUCT:           return %[[VAL_5]] : f32

// STRUCT-LABEL:   func.func @_Z4castCf(
// STRUCT-SAME:                         %[[VAL_0:.*]]: !llvm.struct<(f32, f32)>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_1:.*]] = memref.alloca() : memref<1x!llvm.struct<(f64, f64)>>
// STRUCT:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_3:.*]] = arith.extf %[[VAL_2]] : f32 to f64
// STRUCT:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_5:.*]] = arith.extf %[[VAL_4]] : f32 to f64
// STRUCT:           %[[VAL_6:.*]] = llvm.mlir.undef : !llvm.struct<(f64, f64)>
// STRUCT:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_6]][0] : !llvm.struct<(f64, f64)>
// STRUCT:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_7]][1] : !llvm.struct<(f64, f64)>
// STRUCT:           affine.store %[[VAL_8]], %[[VAL_1]][0] : memref<1x!llvm.struct<(f64, f64)>>
// STRUCT:           %[[VAL_9:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1x!llvm.struct<(f64, f64)>>) -> !llvm.ptr<struct<(f64, f64)>>
// STRUCT:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_9]][0, 0] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// STRUCT:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<f64>
// STRUCT:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_9]][0, 1] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// STRUCT:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<f64>
// STRUCT:           %[[VAL_14:.*]] = arith.addf %[[VAL_11]], %[[VAL_13]] : f64
// STRUCT:           return %[[VAL_14]] : f64

// STRUCT-LABEL:   func.func @_Z12imag_literalv() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f32
// STRUCT:           %[[VAL_1:.*]] = arith.constant 1.000000e+01 : f32
// STRUCT:           %[[VAL_2:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_3]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_4]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           affine.store %[[VAL_5]], %[[VAL_2]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]][0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_6]][0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : f32
// STRUCT:           return %[[VAL_11]] : f32

// STRUCT-LABEL:   func.func @_Z13imag_literal2v() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f32
// STRUCT:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// STRUCT:           %[[VAL_2:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_3]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_4]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           affine.store %[[VAL_5]], %[[VAL_2]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]][0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_6]][0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : f32
// STRUCT:           return %[[VAL_11]] : f32

// STRUCT-LABEL:   func.func @_Z3addv() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = arith.constant 4.000000e+01 : f32
// STRUCT:           %[[VAL_1:.*]] = arith.constant 7.000000e+00 : f32
// STRUCT:           %[[VAL_2:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_3:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_4]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           affine.store %[[VAL_5]], %[[VAL_2]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_6]][0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_6]][0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : f32
// STRUCT:           return %[[VAL_11]] : f32

// STRUCT-LABEL:   func.func @_Z9addassignv() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f32
// STRUCT:           %[[VAL_1:.*]] = arith.constant 3.000000e+01 : f32
// STRUCT:           %[[VAL_2:.*]] = arith.constant 5.000000e+00 : f32
// STRUCT:           %[[VAL_3:.*]] = arith.constant 1.000000e+01 : f32
// STRUCT:           %[[VAL_4:.*]] = memref.alloca() : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_5]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_6]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           affine.store %[[VAL_7]], %[[VAL_4]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_8:.*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr<struct<(f32, f32)>>
// STRUCT:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_8]][0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_11:.*]] = arith.addf %[[VAL_10]], %[[VAL_3]] : f32
// STRUCT:           %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_8]][0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
// STRUCT:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_14:.*]] = arith.addf %[[VAL_13]], %[[VAL_2]] : f32
// STRUCT:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_5]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_15]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           affine.store %[[VAL_16]], %[[VAL_4]][0] : memref<1x!llvm.struct<(f32, f32)>>
// STRUCT:           %[[VAL_17:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_18:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<f32>
// STRUCT:           %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : f32
// STRUCT:           return %[[VAL_19]] : f32

// STRUCT-LABEL:   func.func @_Z3bazv() -> memref<?x!llvm.struct<(struct<(f64, f64)>)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
// STRUCT:           %[[VAL_1:.*]] = arith.constant 3.000000e+01 : f64
// STRUCT:           %[[VAL_2:.*]] = memref.alloc() : memref<1x!llvm.struct<(struct<(f64, f64)>)>>
// STRUCT:           %[[VAL_3:.*]] = memref.cast %[[VAL_2]] : memref<1x!llvm.struct<(struct<(f64, f64)>)>> to memref<?x!llvm.struct<(struct<(f64, f64)>)>>
// STRUCT:           call @_ZN8mcomplexC1Edd(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>)>>, f64, f64) -> ()
// STRUCT:           return %[[VAL_3]] : memref<?x!llvm.struct<(struct<(f64, f64)>)>>

// STRUCT-LABEL:   func.func @_ZN8mcomplexC1Edd(
// STRUCT-SAME:                                 %[[VAL_0:.*]]: memref<?x!llvm.struct<(struct<(f64, f64)>)>>,
// STRUCT-SAME:                                 %[[VAL_1:.*]]: f64,
// STRUCT-SAME:                                 %[[VAL_2:.*]]: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// STRUCT:           %[[VAL_3:.*]] = memref.alloca() : memref<1x!llvm.struct<(f64, f64)>>
// STRUCT:           %[[VAL_4:.*]] = "polygeist.memref2pointer"(%[[VAL_3]]) : (memref<1x!llvm.struct<(f64, f64)>>) -> !llvm.ptr<struct<(f64, f64)>>
// STRUCT:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_4]][0, 0] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// STRUCT:           llvm.store %[[VAL_1]], %[[VAL_5]] : !llvm.ptr<f64>
// STRUCT:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_4]][0, 1] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// STRUCT:           llvm.store %[[VAL_2]], %[[VAL_6]] : !llvm.ptr<f64>
// STRUCT:           %[[VAL_7:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>)>>) -> !llvm.ptr<struct<(struct<(f64, f64)>)>>
// STRUCT:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_7]][0, 0] : (!llvm.ptr<struct<(struct<(f64, f64)>)>>) -> !llvm.ptr<struct<(f64, f64)>>
// STRUCT:           %[[VAL_9:.*]] = affine.load %[[VAL_3]][0] : memref<1x!llvm.struct<(f64, f64)>>
// STRUCT:           llvm.store %[[VAL_9]], %[[VAL_8]] : !llvm.ptr<struct<(f64, f64)>>
// STRUCT:           return
