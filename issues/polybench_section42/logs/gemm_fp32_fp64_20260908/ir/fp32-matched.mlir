#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: f32, %arg5: memref<?x1100xf32>, %arg6: memref<?x1200xf32>, %arg7: memref<?x1100xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg5 : memref<?x1100xf32>
    %1 = bufferization.to_tensor %arg6 : memref<?x1200xf32>
    %2 = bufferization.to_tensor %arg7 : memref<?x1100xf32>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[0, 0] [%5, %3] [1, 1] : tensor<?x1100xf32> to tensor<?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%5, %4] [1, 1] : tensor<?x1200xf32> to tensor<?x?xf32>
    %extracted_slice_1 = tensor.extract_slice %2[0, 0] [%4, %3] [1, 1] : tensor<?x1100xf32> to tensor<?x?xf32>
    %7 = kernel.launch @cublasSgemm_nn_alpha_beta(%extracted_slice_0, %extracted_slice_1, %extracted_slice, %arg4, %arg3) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32, f32) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %7 into %0[0, 0] [%5, %3] [1, 1] : tensor<?x?xf32> into tensor<?x1100xf32>
    %8 = bufferization.to_memref %inserted_slice : memref<?x1100xf32>
    memref.copy %8, %arg5 : memref<?x1100xf32> to memref<?x1100xf32>
    return
  }
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%0.2f \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("C\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("begin dump: %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1000 = arith.constant 1000 : index
    %cst = arith.constant 1.100000e+03 : f32
    %cst_0 = arith.constant 1.200000e+03 : f32
    %cst_1 = arith.constant 1.000000e+03 : f32
    %c20 = arith.constant 20 : index
    %cst_2 = arith.constant 1.500000e+00 : f32
    %cst_3 = arith.constant 1.200000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %c1100_i32 = arith.constant 1100 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %alloc = memref.alloc() : memref<1000x1100xf32>
    affine.for %arg2 = 0 to 1000 {
      %23 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1100 {
        %24 = arith.index_cast %arg3 : index to i32
        %25 = arith.muli %23, %24 : i32
        %26 = arith.addi %25, %c1_i32 : i32
        %27 = arith.remsi %26, %c1000_i32 : i32
        %28 = arith.sitofp %27 : i32 to f32
        %29 = arith.divf %28, %cst_1 : f32
        affine.store %29, %alloc[%arg2, %arg3] : memref<1000x1100xf32>
      }
    }
    affine.for %arg2 = 0 to 1000 {
      affine.for %arg3 = 0 to 1100 {
        %24 = affine.load %alloc[%arg2, %arg3] : memref<1000x1100xf32>
        %25 = arith.mulf %24, %cst_3 : f32
        affine.store %25, %alloc[%arg2, %arg3] : memref<1000x1100xf32>
      }
      %23 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1200 {
        %24 = arith.index_cast %arg3 : index to i32
        %25 = arith.addi %24, %c1_i32 : i32
        %26 = arith.muli %23, %25 : i32
        %27 = arith.remsi %26, %c1200_i32 : i32
        %28 = arith.sitofp %27 : i32 to f32
        %29 = arith.divf %28, %cst_0 : f32
        %30 = arith.mulf %29, %cst_2 : f32
        affine.for %arg4 = 0 to 1100 {
          %31 = arith.index_cast %arg4 : index to i32
          %32 = arith.addi %31, %c2_i32 : i32
          %33 = arith.muli %24, %32 : i32
          %34 = arith.remsi %33, %c1100_i32 : i32
          %35 = arith.sitofp %34 : i32 to f32
          %36 = arith.divf %35, %cst : f32
          %37 = arith.mulf %30, %36 : f32
          %38 = affine.load %alloc[%arg2, %arg4] : memref<1000x1100xf32>
          %39 = arith.addf %38, %37 : f32
          affine.store %39, %alloc[%arg2, %arg4] : memref<1000x1100xf32>
        }
      }
    }
    %0 = llvm.mlir.addressof @stderr : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %4 = llvm.call @fprintf(%1, %3) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %5 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %6 = llvm.mlir.addressof @str1 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %8 = llvm.mlir.addressof @str2 : !llvm.ptr
    %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %10 = llvm.call @fprintf(%5, %7, %9) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %11 = llvm.mlir.addressof @str4 : !llvm.ptr
    %12 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %13 = llvm.mlir.addressof @str3 : !llvm.ptr
    %14 = llvm.getelementptr %13[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    affine.for %arg2 = 0 to 1000 {
      %23 = arith.muli %arg2, %c1000 : index
      affine.for %arg3 = 0 to 1100 {
        %24 = arith.addi %arg3, %23 : index
        %25 = arith.remsi %24, %c20 : index
        %26 = arith.cmpi slt, %25, %c0 : index
        %27 = arith.addi %25, %c20 : index
        %28 = arith.select %26, %27, %25 : index
        %29 = arith.cmpi eq, %28, %c0 : index
        scf.if %29 {
          %34 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
          %35 = llvm.call @fprintf(%34, %14) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        %30 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
        %31 = affine.load %alloc[%arg2, %arg3] : memref<1000x1100xf32>
        %32 = arith.extf %31 : f32 to f64
        %33 = llvm.call @fprintf(%30, %12, %32) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
      }
    }
    %15 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %16 = llvm.mlir.addressof @str5 : !llvm.ptr
    %17 = llvm.getelementptr %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %18 = llvm.call @fprintf(%15, %17, %9) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %19 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %20 = llvm.mlir.addressof @str6 : !llvm.ptr
    %21 = llvm.getelementptr %20[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %22 = llvm.call @fprintf(%19, %21) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    memref.dealloc %alloc : memref<1000x1100xf32>
    return %c0_i32 : i32
  }
}

