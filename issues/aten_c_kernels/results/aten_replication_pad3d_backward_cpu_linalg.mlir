#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (-d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c-2_i32 = arith.constant -2 : i32
    %false = arith.constant false
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 8 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.addi %2, %c-2_i32 : i32
        %4 = arith.cmpi slt, %3, %c0_i32 : i32
        %5 = arith.select %4, %c0_i32, %3 : i32
        %6 = affine.apply #map1(%arg3)
        %7 = arith.cmpi sge, %6, %c0 : index
        %8 = arith.cmpi sge, %3, %c4_i32 : i32
        %9 = arith.select %7, %false, %8 : i1
        %10 = arith.select %9, %c3_i32, %5 : i32
        %11 = arith.addi %1, %10 : i32
        %12 = arith.muli %11, %c5_i32 : i32
        affine.for %arg4 = 0 to 9 {
          %13 = arith.index_cast %arg4 : index to i32
          %14 = arith.addi %13, %c-2_i32 : i32
          %15 = arith.cmpi slt, %14, %c0_i32 : i32
          %16 = arith.select %15, %c0_i32, %14 : i32
          %17 = affine.apply #map1(%arg4)
          %18 = arith.cmpi sge, %17, %c0 : index
          %19 = arith.cmpi sge, %14, %c5_i32 : i32
          %20 = arith.select %18, %false, %19 : i1
          %21 = arith.select %20, %c4_i32, %16 : i32
          %22 = arith.addi %12, %21 : i32
          %23 = arith.muli %22, %c6_i32 : i32
          affine.for %arg5 = 0 to 10 {
            %24 = arith.index_cast %arg5 : index to i32
            %25 = arith.addi %24, %c-2_i32 : i32
            %26 = arith.cmpi slt, %25, %c0_i32 : i32
            %27 = arith.select %26, %c0_i32, %25 : i32
            %28 = affine.apply #map1(%arg5)
            %29 = arith.cmpi sge, %28, %c0 : index
            %30 = arith.cmpi sge, %25, %c6_i32 : i32
            %31 = arith.select %29, %false, %30 : i1
            %32 = arith.select %31, %c5_i32, %27 : i32
            %33 = arith.addi %23, %32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = affine.load %arg0[%arg2 * 720 + %arg5 + %arg3 * 90 + %arg4 * 10] : memref<?xf32>
            %36 = memref.load %arg1[%34] : memref<?xf32>
            %37 = arith.addf %36, %35 : f32
            memref.store %37, %arg1[%34] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}

