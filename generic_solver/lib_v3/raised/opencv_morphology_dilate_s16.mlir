#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_morphology_dilate_s16(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi16>, %arg3: memref<?x?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-32768_i16 = arith.constant -32768 : i16
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to #map()[%0] {
      affine.for %arg5 = 1 to #map()[%1] {
        %2 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %c-32768_i16) -> (i16) {
          %3 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %arg7) -> (i16) {
            %4 = affine.load %arg2[%arg4 + %arg6 - 1, %arg5 + %arg8 - 1] : memref<?x?xi16>
            %5 = arith.extsi %arg9 : i16 to i32
            %6 = arith.extsi %4 : i16 to i32
            %7 = arith.cmpi sgt, %5, %6 : i32
            %8 = arith.select %7, %5, %6 : i32
            %9 = arith.trunci %8 : i32 to i16
            affine.yield %9 : i16
          }
          affine.yield %3 : i16
        }
        affine.store %2, %arg3[%arg4, %arg5] : memref<?x?xi16>
      }
    }
    return
  }
}
