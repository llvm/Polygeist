#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_quick_select_cpu(%arg0: memref<?xf32>, %arg1: i32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = affine.apply #map()[%0]
    %alloca = memref.alloca(%1) : memref<?xi32>
    affine.for %arg3 = 0 to #map()[%0] {
      %3 = arith.index_cast %arg3 : index to i32
      affine.store %3, %alloca[%arg3] : memref<?xi32>
      affine.for %arg4 = #map1(%arg3) to 127 {
        %8 = affine.load %alloca[%arg3] : memref<?xi32>
        %9 = arith.index_cast %arg4 : index to i32
        %10 = affine.load %arg0[%arg4] : memref<?xf32>
        %11 = arith.index_cast %8 : i32 to index
        %12 = memref.load %arg0[%11] : memref<?xf32>
        %13 = arith.cmpf olt, %10, %12 : f32
        %14 = arith.select %13, %9, %8 : i32
        affine.store %14, %alloca[%arg3] : memref<?xi32>
      }
      %4 = affine.load %alloca[%arg3] : memref<?xi32>
      %5 = affine.load %arg0[%arg3] : memref<?xf32>
      %6 = arith.index_cast %4 : i32 to index
      %7 = memref.load %arg0[%6] : memref<?xf32>
      affine.store %7, %arg0[%arg3] : memref<?xf32>
      memref.store %5, %arg0[%6] : memref<?xf32>
    }
    %2 = affine.load %arg0[symbol(%0)] : memref<?xf32>
    affine.store %2, %arg2[0] : memref<?xf32>
    return
  }
}

