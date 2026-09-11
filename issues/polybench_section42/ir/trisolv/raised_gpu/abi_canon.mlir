module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_trisolv(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0 : memref<?xf64>
    %0 = arith.index_cast %dim : index to i32
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<?x?xf64> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %intptr_0 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xf64> -> index
    %3 = arith.index_cast %intptr_0 : index to i64
    %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %intptr_1 = memref.extract_aligned_pointer_as_index %arg2 : memref<?xf64> -> index
    %5 = arith.index_cast %intptr_1 : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    call @polygeist_cublas_pipeline_begin() : () -> ()
    call @polygeist_cublas_dtrsv_lower_row_major(%0, %2, %4, %6) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    call @polygeist_cublas_pipeline_end() : () -> ()
    return
  }
  func.func private @polygeist_cublas_dtrsv_lower_row_major(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @polygeist_cublas_pipeline_begin()
  func.func private @polygeist_cublas_pipeline_end()
}

