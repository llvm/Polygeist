#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.muli %arg2, %c42 : index
      affine.for %arg3 = 0 to 3 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.muli %1, %c6_i32 : i32
        %3 = arith.divsi %2, %c3_i32 : i32
        %4 = arith.addi %1, %c1_i32 : i32
        %5 = arith.muli %4, %c6_i32 : i32
        %6 = arith.addi %5, %c2_i32 : i32
        %7 = arith.divsi %6, %c3_i32 : i32
        %8 = arith.subi %7, %3 : i32
        affine.for %arg4 = 0 to 3 {
          %9 = arith.index_cast %arg4 : index to i32
          %10 = arith.muli %9, %c7_i32 : i32
          %11 = arith.divsi %10, %c3_i32 : i32
          %12 = arith.addi %9, %c1_i32 : i32
          %13 = arith.muli %12, %c7_i32 : i32
          %14 = arith.addi %13, %c2_i32 : i32
          %15 = arith.divsi %14, %c3_i32 : i32
          %16 = arith.subi %15, %11 : i32
          %17 = arith.muli %8, %16 : i32
          %18 = arith.sitofp %17 : i32 to f32
          %19 = arith.muli %arg4, %c7 : index
          %20 = arith.cmpi slt, %19, %c0 : index
          %21 = arith.subi %c-1, %19 : index
          %22 = arith.select %20, %21, %19 : index
          %23 = arith.divsi %22, %c3 : index
          %24 = arith.subi %c-1, %23 : index
          %25 = arith.select %20, %24, %23 : index
          %26 = arith.addi %25, %c3 : index
          affine.for %arg5 = #map1(%arg3) to #map2(%arg3) {
            %27 = arith.muli %arg5, %c7 : index
            scf.for %arg6 = %25 to %26 step %c1 {
              %28 = affine.load %arg0[%arg4 + %arg2 * 9 + %arg3 * 3] : memref<?xf32>
              %29 = arith.divf %28, %18 : f32
              %30 = arith.addi %arg6, %0 : index
              %31 = arith.addi %30, %27 : index
              %32 = memref.load %arg1[%31] : memref<?xf32>
              %33 = arith.addf %32, %29 : f32
              memref.store %33, %arg1[%31] : memref<?xf32>
            }
          }
        }
      }
    }
    return
  }
}

