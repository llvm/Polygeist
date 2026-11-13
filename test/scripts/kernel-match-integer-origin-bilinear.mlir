// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --enable-structured-rewrite > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null
// RUN: sed 's/5.000000e-01/2.500000e-01/g' %s > %t.weight.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.weight.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/cmpi slt/cmpi sle/g' %s > %t.clamp.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.clamp.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/f32/f64/g' %s > %t.type.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.type.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NEGATIVE
// RUN: sed 's/grid_interpolation/another_function/g' %s > %t.rename.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.rename.mlir --enable-structured-rewrite | sed '/^\/\/ CHECK/d' | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @grid_interpolation(%arg0: memref<?x3x7x7xf32>, %arg1: memref<?x3x14x14xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4096 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 14 : index
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c4_i32 = arith.constant 7 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = polygeist.submap(%arg1, %c4096, %c3, %c8, %c8) {map = #map} : (memref<?x3x14x14xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %1 = linalg.index 0 : index
      %2 = linalg.index 1 : index
      %3 = linalg.index 2 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = arith.remsi %4, %c2_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_0 : f32
      %8 = arith.subf %cst, %7 : f32
      %9 = arith.divsi %4, %c2_i32 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.addi %9, %c1_i32 : i32
      %12 = arith.cmpi slt, %11, %c4_i32 : i32
      %13 = arith.select %12, %11, %9 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.cmpi slt, %3, %c0 : index
      %16 = arith.subi %c-1, %3 : index
      %17 = arith.select %15, %16, %3 : index
      %18 = arith.divsi %17, %c2 : index
      %19 = arith.subi %c-1, %18 : index
      %20 = arith.select %15, %19, %18 : index
      %21 = linalg.index 3 : index
      %22 = arith.index_cast %21 : index to i32
      %23 = arith.remsi %22, %c2_i32 : i32
      %24 = arith.sitofp %23 : i32 to f32
      %25 = arith.mulf %24, %cst_0 : f32
      %26 = arith.subf %cst, %25 : f32
      %27 = arith.divsi %22, %c2_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.cmpi slt, %21, %c0 : index
      %30 = arith.subi %c-1, %21 : index
      %31 = arith.select %29, %30, %21 : index
      %32 = arith.divsi %31, %c2 : index
      %33 = arith.subi %c-1, %32 : index
      %34 = arith.select %29, %33, %32 : index
      %35 = memref.load %arg0[%1, %2, %20, %34] : memref<?x3x7x7xf32>
      %36 = arith.mulf %26, %35 : f32
      %37 = arith.addi %27, %c1_i32 : i32
      %38 = arith.cmpi slt, %37, %c4_i32 : i32
      %39 = arith.select %38, %37, %27 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = memref.load %arg0[%1, %2, %10, %40] : memref<?x3x7x7xf32>
      %42 = arith.mulf %25, %41 : f32
      %43 = arith.addf %36, %42 : f32
      %44 = arith.mulf %8, %43 : f32
      %45 = memref.load %arg0[%1, %2, %14, %28] : memref<?x3x7x7xf32>
      %46 = arith.mulf %26, %45 : f32
      %47 = memref.load %arg0[%1, %2, %14, %40] : memref<?x3x7x7xf32>
      %48 = arith.mulf %25, %47 : f32
      %49 = arith.addf %46, %48 : f32
      %50 = arith.mulf %7, %49 : f32
      %51 = arith.addf %44, %50 : f32
      linalg.yield %51 : f32
    }
    return
  }
}


// CHECK: kernel.launch @cudnnAdaptivePool_f32_bilinear_integer_origin_flat2
// CHECK-NOT: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnAdaptivePool_f32_bilinear_integer_origin_flat2
// CHECK-NEGATIVE: linalg.generic
// CHECK-NEGATIVE-NOT: kernel.launch @cudnnAdaptivePool_f32_bilinear_integer_origin_flat2
