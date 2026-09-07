module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %dim = memref.dim %arg5, %c0 : memref<?xf64>
    %1 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %2 = arith.index_cast %intptr : index to i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_1d(%1, %3) : (i32, !llvm.ptr) -> ()
    %dim_1 = memref.dim %arg7, %c0 : memref<?xf64>
    %4 = arith.index_cast %dim_1 : index to i32
    %intptr_2 = memref.extract_aligned_pointer_as_index %arg7 : memref<?xf64> -> index
    %5 = arith.index_cast %intptr_2 : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    call @polygeist_cublas_memset_zero_1d(%4, %6) : (i32, !llvm.ptr) -> ()
    %dim_3 = memref.dim %arg3, %c1 : memref<?x?xf64>
    %7 = arith.index_cast %dim_3 : index to i32
    %intptr_4 = memref.extract_aligned_pointer_as_index %arg3 : memref<?x?xf64> -> index
    %8 = arith.index_cast %intptr_4 : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %intptr_5 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %10 = arith.index_cast %intptr_5 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %subview = memref.subview %arg5[0] [%0] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_6 = memref.extract_aligned_pointer_as_index %subview : memref<?xf64, strided<[1]>> -> index
    %12 = arith.index_cast %intptr_6 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %9, %7, %11, %cst_0, %13) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_7 = memref.dim %arg4, %c1 : memref<?x?xf64>
    %14 = arith.index_cast %dim_7 : index to i32
    %intptr_8 = memref.extract_aligned_pointer_as_index %arg4 : memref<?x?xf64> -> index
    %15 = arith.index_cast %intptr_8 : index to i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %arg6 : memref<?xf64> -> index
    %17 = arith.index_cast %intptr_9 : index to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %subview_10 = memref.subview %arg7[0] [%0] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
    %intptr_11 = memref.extract_aligned_pointer_as_index %subview_10 : memref<?xf64, strided<[1]>> -> index
    %19 = arith.index_cast %intptr_11 : index to i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    call @polygeist_cublas_dgemv(%arg0, %arg0, %cst, %16, %14, %18, %cst_0, %20) : (i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr) -> ()
    %dim_12 = memref.dim %arg7, %c0 : memref<?xf64>
    %21 = arith.index_cast %dim_12 : index to i32
    %intptr_13 = memref.extract_aligned_pointer_as_index %arg5 : memref<?xf64> -> index
    %22 = arith.index_cast %intptr_13 : index to i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %intptr_14 = memref.extract_aligned_pointer_as_index %arg7 : memref<?xf64> -> index
    %24 = arith.index_cast %intptr_14 : index to i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    call @polygeist_cublas_daxpby(%21, %arg1, %23, %arg2, %25) : (i32, f64, !llvm.ptr, f64, !llvm.ptr) -> ()
    return
  }
  func.func private @polygeist_cublas_memset_zero_1d(i32, !llvm.ptr)
  func.func private @polygeist_cublas_dgemv(i32, i32, f64, !llvm.ptr, i32, !llvm.ptr, f64, !llvm.ptr)
  func.func private @polygeist_cublas_daxpby(i32, f64, !llvm.ptr, f64, !llvm.ptr)
}

