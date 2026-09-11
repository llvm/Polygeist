#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dyn_quant_pack_4bit_weight_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x32xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c48 = arith.constant 48 : index
    %c63 = arith.constant 63 : index
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %false = arith.constant false
    %c4_i32 = arith.constant 4 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 1.500000e+01 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca(%c48) : memref<?xf32>
    %alloca_1 = memref.alloca(%c48) : memref<?xf32>
    affine.for %arg4 = 0 to 48 {
      %0 = affine.load %arg0[%arg4, 0] : memref<?x64xf32>
      affine.store %0, %alloca[%arg4] : memref<?xf32>
      affine.store %0, %alloca_1[%arg4] : memref<?xf32>
      %subview = memref.subview %arg0[%arg4, 1] [1, %c63] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_2 = memref.subview %alloca[%arg4] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      %subview_3 = memref.subview %alloca_1[%arg4] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_2, %subview_3 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %out: f32, %out_4: f32):
        %7 = arith.cmpf olt, %in, %out_4 : f32
        %8 = arith.select %7, %in, %out_4 : f32
        %9 = arith.cmpf ogt, %in, %out : f32
        %10 = arith.select %9, %in, %out : f32
        linalg.yield %10, %8 : f32, f32
      }
      %1 = affine.load %alloca[%arg4] : memref<?xf32>
      %2 = affine.load %alloca_1[%arg4] : memref<?xf32>
      %3 = arith.subf %1, %2 : f32
      %4 = arith.divf %3, %cst_0 : f32
      affine.store %4, %arg2[%arg4] : memref<?xf32>
      %5 = arith.negf %2 : f32
      %6 = arith.divf %5, %4 : f32
      affine.store %6, %arg3[%arg4] : memref<?xf32>
      affine.for %arg5 = 0 to 64 step 2 {
        %7 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
        %8 = affine.load %arg2[%arg4] : memref<?xf32>
        %9 = arith.divf %7, %8 : f32
        %10 = affine.load %arg3[%arg4] : memref<?xf32>
        %11 = arith.addf %9, %10 : f32
        %12 = arith.addf %11, %cst : f32
        %13 = arith.fptosi %12 : f32 to i32
        %14 = affine.load %arg0[%arg4, %arg5 + 1] : memref<?x64xf32>
        %15 = arith.divf %14, %8 : f32
        %16 = arith.addf %15, %10 : f32
        %17 = arith.addf %16, %cst : f32
        %18 = arith.fptosi %17 : f32 to i32
        %19 = arith.cmpi slt, %13, %c0_i32 : i32
        %20 = arith.select %19, %c0_i32, %13 : i32
        %21 = arith.cmpi sgt, %13, %c15_i32 : i32
        %22 = arith.select %19, %false, %21 : i1
        %23 = arith.select %22, %c15_i32, %20 : i32
        %24 = arith.cmpi slt, %18, %c0_i32 : i32
        %25 = arith.select %24, %c0_i32, %18 : i32
        %26 = arith.cmpi sgt, %18, %c15_i32 : i32
        %27 = arith.select %24, %false, %26 : i1
        %28 = arith.select %27, %c15_i32, %25 : i32
        %29 = arith.shli %28, %c4_i32 : i32
        %30 = arith.ori %23, %29 : i32
        %31 = arith.trunci %30 : i32 to i8
        %32 = arith.cmpi slt, %arg5, %c0 : index
        %33 = arith.subi %c-1, %arg5 : index
        %34 = arith.select %32, %33, %arg5 : index
        %35 = arith.divsi %34, %c2 : index
        %36 = arith.subi %c-1, %35 : index
        %37 = arith.select %32, %36, %35 : index
        memref.store %31, %arg1[%arg4, %37] : memref<?x32xi8>
      }
    }
    return
  }
}

