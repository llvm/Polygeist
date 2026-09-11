#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_logcumsumexp_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca(%c32) : memref<?xf32>
    affine.for %arg2 = 0 to 32 {
      %0 = affine.load %arg0[%arg2, 0] : memref<?x64xf32>
      affine.store %0, %arg1[%arg2, 0] : memref<?x64xf32>
      affine.store %0, %alloca[%arg2] : memref<?xf32>
      %subview = memref.subview %arg0[%arg2, 1] [1, %c63] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, 1] [1, %c63] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %alloca[%arg2] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_0, %subview_1 : memref<?xf32, strided<[1], offset: ?>>, memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %out: f32, %out_2: f32):
        %1 = arith.cmpf ogt, %out_2, %in : f32
        %2 = arith.select %1, %out_2, %in : f32
        %3 = arith.subf %out_2, %in : f32
        %4 = arith.cmpf olt, %3, %cst : f32
        %5 = arith.negf %3 : f32
        %6 = arith.select %4, %5, %3 : f32
        %7 = arith.negf %6 : f32
        %8 = math.exp %7 : f32
        %9 = math.log1p %8 : f32
        %10 = arith.addf %2, %9 : f32
        linalg.yield %10, %10 : f32, f32
      }
    }
    return
  }
  func.func private @log1pf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

