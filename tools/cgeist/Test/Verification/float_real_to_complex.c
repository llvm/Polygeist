// RUN: cgeist %s --struct-abi=0 --function='*' -S | FileCheck %s --check-prefix=STRUCT
// COM: we dont support this yet: cgeist %s --function='*' -S | FileCheck %s

__complex__ float complextest()
{
  __complex__ float  z=8.0;
  return z;

}

// STRUCT-LABEL:   func.func @complextest() -> !llvm.struct<(f32, f32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// STRUCT:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// STRUCT:           %[[VAL_1:.*]] = arith.constant 8.000000e+00 : f32
// STRUCT:           %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_3:.*]] = llvm.insertvalue  %[[VAL_1]], %[[VAL_2]][0] : !llvm.struct<(f32, f32)>
// STRUCT:           %[[VAL_4:.*]] = llvm.insertvalue  %[[VAL_0]], %[[VAL_3]][1] : !llvm.struct<(f32, f32)>
// STRUCT:           return %[[VAL_4]] : !llvm.struct<(f32, f32)>