#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg7 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg8 restrict : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg9 restrict : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg10 restrict : memref<?x?xf64>
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg0 : i32 to index
    %9 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%0 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?xf64>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%8, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%5, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %9[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_1 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %15 = arith.mulf %arg4, %in : f64
      %16 = arith.mulf %15, %in_5 : f64
      %17 = arith.addf %out, %16 : f64
      linalg.yield %17 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %10 into %9[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %11 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %11, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    %12 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%4 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %15 = arith.mulf %out, %arg5 : f64
      linalg.yield %15 : f64
    } -> tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %3[0, 0] [%7, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %12[0, 0] [%8, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%10, %extracted_slice_2 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_3 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %15 = arith.mulf %in, %in_5 : f64
      %16 = arith.addf %out, %15 : f64
      linalg.yield %16 : f64
    } -> tensor<?x?xf64>
    %inserted_slice_4 = tensor.insert_slice %13 into %12[0, 0] [%8, %6] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %14 = bufferization.to_memref %inserted_slice_4 : memref<?x?xf64>
    memref.copy %14, %arg10 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%0.2lf \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("D\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("begin dump: %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c800 = arith.constant 800 : index
    %cst = arith.constant 1.100000e+03 : f64
    %cst_0 = arith.constant 1.200000e+03 : f64
    %cst_1 = arith.constant 9.000000e+02 : f64
    %cst_2 = arith.constant 8.000000e+02 : f64
    %c20 = arith.constant 20 : index
    %cst_3 = arith.constant 0.000000e+00 : f64
    %cst_4 = arith.constant 1.500000e+00 : f64
    %cst_5 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %c1100_i32 = arith.constant 1100 : i32
    %c900_i32 = arith.constant 900 : i32
    %c800_i32 = arith.constant 800 : i32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<800x900xf64>
    %alloc_6 = memref.alloc() : memref<800x1100xf64>
    %alloc_7 = memref.alloc() : memref<1100x900xf64>
    %alloc_8 = memref.alloc() : memref<900x1200xf64>
    %alloc_9 = memref.alloc() : memref<800x1200xf64>
    %0 = "polygeist.subindex"(%alloc_6, %c0) : (memref<800x1100xf64>, index) -> memref<?xf64>
    %1 = "polygeist.memref2pointer"(%0) : (memref<?xf64>) -> !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x?xf64>
    %3 = "polygeist.subindex"(%alloc_7, %c0) : (memref<1100x900xf64>, index) -> memref<?xf64>
    %4 = "polygeist.memref2pointer"(%3) : (memref<?xf64>) -> !llvm.ptr
    %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr) -> memref<?x?xf64>
    %6 = "polygeist.subindex"(%alloc_8, %c0) : (memref<900x1200xf64>, index) -> memref<?xf64>
    %7 = "polygeist.memref2pointer"(%6) : (memref<?xf64>) -> !llvm.ptr
    %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x?xf64>
    %9 = "polygeist.subindex"(%alloc_9, %c0) : (memref<800x1200xf64>, index) -> memref<?xf64>
    %10 = "polygeist.memref2pointer"(%9) : (memref<?xf64>) -> !llvm.ptr
    %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?x?xf64>
    affine.for %arg2 = 0 to 800 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1100 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.muli %38, %39 : i32
        %41 = arith.addi %40, %c1_i32 : i32
        %42 = arith.remsi %41, %c800_i32 : i32
        %43 = arith.sitofp %42 : i32 to f64
        %44 = arith.divf %43, %cst_2 : f64
        affine.store %44, %2[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    affine.for %arg2 = 0 to 1100 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 900 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.addi %39, %c1_i32 : i32
        %41 = arith.muli %38, %40 : i32
        %42 = arith.remsi %41, %c900_i32 : i32
        %43 = arith.sitofp %42 : i32 to f64
        %44 = arith.divf %43, %cst_1 : f64
        affine.store %44, %5[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    affine.for %arg2 = 0 to 900 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1200 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.addi %39, %c3_i32 : i32
        %41 = arith.muli %38, %40 : i32
        %42 = arith.addi %41, %c1_i32 : i32
        %43 = arith.remsi %42, %c1200_i32 : i32
        %44 = arith.sitofp %43 : i32 to f64
        %45 = arith.divf %44, %cst_0 : f64
        affine.store %45, %8[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    affine.for %arg2 = 0 to 800 {
      %38 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1200 {
        %39 = arith.index_cast %arg3 : index to i32
        %40 = arith.addi %39, %c2_i32 : i32
        %41 = arith.muli %38, %40 : i32
        %42 = arith.remsi %41, %c1100_i32 : i32
        %43 = arith.sitofp %42 : i32 to f64
        %44 = arith.divf %43, %cst : f64
        affine.store %44, %11[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    %12 = "polygeist.subindex"(%alloc, %c0) : (memref<800x900xf64>, index) -> memref<?xf64>
    %13 = "polygeist.memref2pointer"(%12) : (memref<?xf64>) -> !llvm.ptr
    %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?x?xf64>
    affine.for %arg2 = 0 to 800 {
      affine.for %arg3 = 0 to 900 {
        affine.store %cst_3, %14[%arg2, %arg3] : memref<?x?xf64>
        affine.for %arg4 = 0 to 1100 {
          %38 = affine.load %2[%arg2, %arg4] : memref<?x?xf64>
          %39 = arith.mulf %38, %cst_4 : f64
          %40 = affine.load %5[%arg4, %arg3] : memref<?x?xf64>
          %41 = arith.mulf %39, %40 : f64
          %42 = affine.load %14[%arg2, %arg3] : memref<?x?xf64>
          %43 = arith.addf %42, %41 : f64
          affine.store %43, %14[%arg2, %arg3] : memref<?x?xf64>
        }
      }
    }
    affine.for %arg2 = 0 to 800 {
      affine.for %arg3 = 0 to 1200 {
        %38 = affine.load %11[%arg2, %arg3] : memref<?x?xf64>
        %39 = arith.mulf %38, %cst_5 : f64
        affine.store %39, %11[%arg2, %arg3] : memref<?x?xf64>
        affine.for %arg4 = 0 to 900 {
          %40 = affine.load %14[%arg2, %arg4] : memref<?x?xf64>
          %41 = affine.load %8[%arg4, %arg3] : memref<?x?xf64>
          %42 = arith.mulf %40, %41 : f64
          %43 = affine.load %11[%arg2, %arg3] : memref<?x?xf64>
          %44 = arith.addf %43, %42 : f64
          affine.store %44, %11[%arg2, %arg3] : memref<?x?xf64>
        }
      }
    }
    %15 = llvm.mlir.addressof @stderr : !llvm.ptr
    %16 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %17 = llvm.mlir.addressof @str0 : !llvm.ptr
    %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %19 = llvm.call @fprintf(%16, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %20 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %21 = llvm.mlir.addressof @str1 : !llvm.ptr
    %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %23 = llvm.mlir.addressof @str2 : !llvm.ptr
    %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %25 = llvm.call @fprintf(%20, %22, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %26 = llvm.mlir.addressof @str4 : !llvm.ptr
    %27 = llvm.getelementptr %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %28 = llvm.mlir.addressof @str3 : !llvm.ptr
    %29 = llvm.getelementptr %28[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    affine.for %arg2 = 0 to 800 {
      %38 = arith.muli %arg2, %c800 : index
      affine.for %arg3 = 0 to 1200 {
        %39 = arith.addi %arg3, %38 : index
        %40 = arith.remsi %39, %c20 : index
        %41 = arith.cmpi slt, %40, %c0 : index
        %42 = arith.addi %40, %c20 : index
        %43 = arith.select %41, %42, %40 : index
        %44 = arith.cmpi eq, %43, %c0 : index
        scf.if %44 {
          %48 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
          %49 = llvm.call @fprintf(%48, %29) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        %45 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
        %46 = affine.load %11[%arg2, %arg3] : memref<?x?xf64>
        %47 = llvm.call @fprintf(%45, %27, %46) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
      }
    }
    %30 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %31 = llvm.mlir.addressof @str5 : !llvm.ptr
    %32 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %33 = llvm.call @fprintf(%30, %32, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %34 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    %35 = llvm.mlir.addressof @str6 : !llvm.ptr
    %36 = llvm.getelementptr %35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %37 = llvm.call @fprintf(%34, %36) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    memref.dealloc %alloc : memref<800x900xf64>
    memref.dealloc %alloc_6 : memref<800x1100xf64>
    memref.dealloc %alloc_7 : memref<1100x900xf64>
    memref.dealloc %alloc_8 : memref<900x1200xf64>
    memref.dealloc %alloc_9 : memref<800x1200xf64>
    return %c0_i32 : i32
  }
}

