#map = affine_map<(d0)[s0] -> (s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c6 = arith.constant 6 : index
    %c42 = arith.constant 42 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg2 = 0 to 2 {
      %0 = arith.muli %arg2, %c42 : index
      affine.for %arg3 = 0 to 3 {
        %alloca = memref.alloca(%c3) : memref<?xi32>
        %alloca_0 = memref.alloca(%c3) : memref<?xf32>
        affine.for %arg4 = 0 to 3 {
          %1 = arith.index_cast %arg4 : index to i32
          %2 = arith.muli %1, %c7_i32 : i32
          %3 = arith.divsi %2, %c3_i32 : i32
          %4 = arith.addi %1, %c1_i32 : i32
          %5 = arith.muli %4, %c7_i32 : i32
          %6 = arith.addi %5, %c2_i32 : i32
          %7 = arith.divsi %6, %c3_i32 : i32
          %8 = arith.index_cast %7 : i32 to index
          %9 = arith.index_cast %3 : i32 to index
          %10 = arith.subi %8, %9 : index
          %11 = arith.muli %arg4, %c7 : index
          %12 = arith.cmpi slt, %11, %c0 : index
          %13 = arith.subi %c-1, %11 : index
          %14 = arith.select %12, %13, %11 : index
          %15 = arith.divsi %14, %c3 : index
          %16 = arith.subi %c-1, %15 : index
          %17 = arith.select %12, %16, %15 : index
          %18 = arith.addi %17, %c3 : index
          affine.store %c0_i32, %alloca[%arg4] : memref<?xi32>
          affine.store %cst, %alloca_0[%arg4] : memref<?xf32>
          %19 = polygeist.submap(%alloca, %arg4, %c6) {map = #map} : (memref<?xi32>, index, index) -> memref<?xi32>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["reduction"]} outs(%19 : memref<?xi32>) {
          ^bb0(%out: i32):
            %24 = arith.index_cast %out : i32 to index
            %25 = arith.addi %24, %10 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = linalg.index 0 : index
            %28 = affine.apply #map2(%arg3)
            %29 = arith.cmpi sge, %27, %28 : index
            %30 = affine.apply #map3(%arg3)
            %31 = arith.cmpi slt, %27, %30 : index
            %32 = arith.andi %29, %31 : i1
            %33 = arith.select %32, %26, %out : i32
            linalg.yield %33 : i32
          }
          affine.for %arg5 = #map2(%arg3) to #map3(%arg3) {
            %24 = affine.load %alloca_0[%arg4] : memref<?xf32>
            %25 = arith.muli %arg5, %c7 : index
            %26 = scf.for %arg6 = %17 to %18 step %c1 iter_args(%arg7 = %24) -> (f32) {
              %27 = arith.addi %arg6, %0 : index
              %28 = arith.addi %27, %25 : index
              %29 = memref.load %arg0[%28] : memref<?xf32>
              %30 = arith.addf %arg7, %29 : f32
              scf.yield %30 : f32
            }
            affine.store %26, %alloca_0[%arg4] : memref<?xf32>
          }
          %20 = affine.load %alloca[%arg4] : memref<?xi32>
          %21 = affine.load %alloca_0[%arg4] : memref<?xf32>
          %22 = arith.sitofp %20 : i32 to f32
          %23 = arith.divf %21, %22 : f32
          affine.store %23, %arg1[%arg4 + %arg2 * 9 + %arg3 * 3] : memref<?xf32>
        }
      }
    }
    return
  }
}

