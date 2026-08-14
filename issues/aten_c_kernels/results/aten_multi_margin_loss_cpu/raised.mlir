#map = affine_map<(d0)[s0] -> (s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multi_margin_loss_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.600000e+01 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    %alloca = memref.alloca(%c32) : memref<?xf32>
    affine.for %arg6 = 0 to 32 {
      %1 = affine.load %arg1[%arg6] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      affine.store %cst, %alloca[%arg6] : memref<?xf32>
      %3 = polygeist.submap(%alloca, %arg6, %c16) {map = #map} : (memref<?xf32>, index, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["reduction"]} outs(%3 : memref<?xf32>) {
      ^bb0(%out: f32):
        %8 = linalg.index 0 : index
        %9 = arith.index_cast %8 : index to i32
        %10 = arith.cmpi ne, %9, %1 : i32
        %11 = scf.if %10 -> (f32) {
          %12 = memref.load %arg0[%arg6, %2] : memref<?x16xf32>
          %13 = arith.subf %arg3, %12 : f32
          %14 = memref.load %arg0[%arg6, %8] : memref<?x16xf32>
          %15 = arith.addf %13, %14 : f32
          %16 = arith.cmpf ogt, %15, %cst : f32
          %17 = scf.if %16 -> (f32) {
            %18 = affine.apply #map2()[%0]
            %19 = arith.cmpi eq, %18, %c0 : index
            %20 = arith.mulf %15, %15 : f32
            %21 = arith.select %19, %15, %20 : f32
            %22 = arith.addf %out, %21 : f32
            scf.yield %22 : f32
          } else {
            scf.yield %out : f32
          }
          scf.yield %17 : f32
        } else {
          scf.yield %out : f32
        }
        linalg.yield %11 : f32
      }
      %4 = affine.load %alloca[%arg6] : memref<?xf32>
      %5 = memref.load %arg2[%2] : memref<?xf32>
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.divf %6, %cst_0 : f32
      affine.store %7, %arg5[%arg6] : memref<?xf32>
    }
    return
  }
}

