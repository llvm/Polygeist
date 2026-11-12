#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @scipy_csc_matvec_beta0_i64(%arg0: i64, %arg1: i64, %arg2: memref<?xi64>, %arg3: memref<?xi64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = polygeist.submap(%arg6, %0) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : memref<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    }
    %2 = arith.index_cast %arg1 : i64 to index
    affine.for %arg7 = 0 to %2 {
      %3 = affine.load %arg2[%arg7] : memref<?xi64>
      %4 = affine.load %arg2[%arg7 + 1] : memref<?xi64>
      %5 = arith.index_cast %4 : i64 to index
      %6 = arith.index_cast %3 : i64 to index
      scf.for %arg8 = %6 to %5 step %c1 {
        %7 = memref.load %arg3[%arg8] : memref<?xi64>
        %8 = arith.index_cast %7 : i64 to index
        %9 = memref.load %arg4[%arg8] : memref<?xf64>
        %10 = affine.load %arg5[%arg7] : memref<?xf64>
        %11 = arith.mulf %9, %10 : f64
        %12 = memref.load %arg6[%8] : memref<?xf64>
        %13 = arith.addf %12, %11 : f64
        memref.store %13, %arg6[%8] : memref<?xf64>
      }
    }
    return
  }
}

