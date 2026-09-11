// A generated GPU kernel exported as a plain device-pointer wrapper.  The
// wrapper intentionally contains no graph management: its caller can compose
// this launch with cuBLAS/cuDNN/cuTENSOR calls inside one Polygeist CUDA Graph.
module attributes {gpu.container_module} {
  gpu.module @generated_kernels {
    gpu.func @scale_kernel(%data: !llvm.ptr, %n: i64, %scale: f64) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %block_base = arith.muli %bid, %bdim : index
      %idx = arith.addi %block_base, %tid : index
      %idx64 = arith.index_cast %idx : index to i64
      %in_bounds = arith.cmpi ult, %idx64, %n : i64
      scf.if %in_bounds {
        %element = llvm.getelementptr %data[%idx64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %old = llvm.load %element : !llvm.ptr -> f64
        %new = arith.mulf %old, %scale : f64
        llvm.store %new, %element : f64, !llvm.ptr
      }
      gpu.return
    }
  }

  func.func @polygeist_generated_scale_f64(%data: !llvm.ptr, %n: i64,
                                            %scale: f64) attributes {
      llvm.emit_c_interface
  } {
    %c1 = arith.constant 1 : index
    %c255 = arith.constant 255 : i64
    %c256 = arith.constant 256 : index
    %c256_i64 = arith.constant 256 : i64
    %n_plus = arith.addi %n, %c255 : i64
    %blocks64 = arith.divui %n_plus, %c256_i64 : i64
    %blocks = arith.index_cast %blocks64 : i64 to index
    gpu.launch_func @generated_kernels::@scale_kernel
        blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%data : !llvm.ptr, %n : i64, %scale : f64)
        {polygeist.cuda_graph_safe}
    return
  }
}
