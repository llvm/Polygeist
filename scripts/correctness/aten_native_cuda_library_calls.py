#!/usr/bin/env python3
"""Audit external CUDA-library use by the native counterparts of ATen fixtures.

This is intentionally separate from aten_cuda_library_audit.py: that audit asks
which NVIDIA API could replace an extracted kernel, while this audit asks what
the pinned PyTorch CUDA implementation actually invokes.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues/aten_c_kernels"
PYTORCH = ROOT / "third_party/pytorch"
OUTPUT = CORPUS / "native_cuda_external_library_calls.csv"
REPORT = CORPUS / "NATIVE_CUDA_EXTERNAL_LIBRARY_CALLS.md"


def external(libraries: str, form: str, source: str, symbol: str, note: str):
    return libraries, form, source, symbol, note


# Cases whose library path crosses translation units, backend selection, or a
# composite ATen implementation and therefore cannot be recovered from the
# CPU/CUDA dispatch-stub registration alone. Each entry is backed by the pinned
# PyTorch source named in the row.
OVERRIDES = {
    # Dense BLAS and BLAS-based convolution fallbacks.
    "aten_addmm": external("cuBLAS", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/cuda/Blas.cpp", "at::cuda::blas::gemm", "CUDA addmm lowers to GEMM/GEMM-and-bias after layout preparation."),
    "aten_bmm": external("cuBLAS", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/cuda/Blas.cpp", "at::cuda::blas::bgemm", "CUDA bmm uses batched GEMM for the supported dense layouts."),
    "aten_dot": external("cuBLAS", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/cuda/Blas.cpp", "at::cuda::blas::dot", "Dense CUDA dot dispatches to the cuBLAS dot wrapper."),
    "aten_mm": external("cuBLAS", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/cuda/Blas.cpp", "at::cuda::blas::gemm", "Dense CUDA mm dispatches to GEMM."),
    "aten_mv": external("cuBLAS", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/cuda/Blas.cpp", "at::cuda::blas::gemv", "Dense CUDA mv dispatches to GEMV."),
    "aten_int_mm_out_cpu": external("cuBLAS", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/cuda/Blas.cpp", "at::cuda::blas::int8_gemm", "The CUDA int8 matrix product uses the cuBLAS int8 GEMM wrapper."),
    "aten_conv2d": external("cuDNN|cuBLAS", "CONDITIONAL_BACKEND", "aten/src/ATen/native/Convolution.cpp|aten/src/ATen/native/cuda/ConvolutionMM2d.cu", "cudnn_convolution|at::cuda::blas::gemm", "Public convolution may select cuDNN; the explicit slow CUDA 2-D path is im2col plus cuBLAS GEMM."),
    "aten_conv1d": external("cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/Convolution.cpp", "cudnn_convolution", "The public convolution dispatcher can select the cuDNN backend when its guards hold."),
    "aten_conv3d": external("cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/Convolution.cpp", "cudnn_convolution", "The public convolution dispatcher can select the cuDNN backend when its guards hold."),
    "aten_conv_transpose2d": external("cuBLAS", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/NaiveConvolutionTranspose2d.cu", "at::cuda::blas::gemm", "The explicit slow CUDA transpose-convolution path combines GEMM with ATen column kernels."),
    "aten_conv_transpose3d_cpu": external("cuBLAS", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu", "at::cuda::blas::gemm", "The CUDA counterpart combines GEMM with ATen volume-column kernels."),
    "aten_conv_transpose3d_backward_cpu": external("cuBLAS", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu", "at::cuda::blas::gemm", "The CUDA backward path combines GEMM with ATen volume-column kernels."),
    "aten_conv_transpose3d_grad_weight_cpu": external("cuBLAS", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu", "at::cuda::blas::gemm", "The CUDA weight-gradient path combines GEMM with ATen volume-column kernels."),
    "aten_dilated_convolution_cpu": external("cuBLAS", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/NaiveDilatedConvolution.cu", "at::cuda::blas::gemm|gemv", "The explicit dilated CUDA fallback uses BLAS plus ATen unfold/fold kernels."),
    "aten_conv_tbc_cpu": external("cuBLAS", "ATEN_COMPOSITION", "aten/src/ATen/native/ConvolutionTBC.cpp", "Tensor::addmm_", "CompositeExplicitAutograd conv_tbc invokes addmm, whose CUDA implementation uses cuBLAS."),
    "aten_conv_tbc_backward_cpu": external("cuBLAS", "ATEN_COMPOSITION", "aten/src/ATen/native/ConvolutionTBC.cpp", "Tensor::addmm_", "CompositeExplicitAutograd backward invokes addmm, whose CUDA implementation uses cuBLAS."),

    # cuDNN-selectable public operations.
    "aten_batch_norm": external("cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/Normalization.cpp|aten/src/ATen/native/cudnn/BatchNorm.cpp", "cudnn_batch_norm", "The high-level batch-normalization dispatcher may select cuDNN; native CUDA kernels remain fallback paths."),
    "aten_batch_norm_cpu_entry": external("cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/Normalization.cpp|aten/src/ATen/native/cudnn/BatchNorm.cpp", "cudnn_batch_norm", "The benchmarked high-level batch-normalization route may select cuDNN; this is not true of every extracted internal stage."),
    "aten_ctc_loss_cpu": external("cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/LossCTC.cpp|aten/src/ATen/native/cudnn/LossCTC.cpp", "cudnn_ctc_loss", "ATen selects cuDNN only for the supported CTC shapes, dtypes, and determinism constraints."),
    "aten_ctc_loss_backward_cpu": external("cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/LossCTC.cpp|aten/src/ATen/native/cudnn/LossCTC.cpp", "cudnn_ctc_loss", "The cuDNN CTC route jointly computes loss and gradient; the native CUDA fallback is separate."),

    # Sparse BLAS.
    "aten_sparse_addmm_cpu": external("cuSPARSE", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu|aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp", "cusparseSpMM", "COO metadata is converted/prepared by ATen before the CSR sparse-dense multiply."),
    "aten_hspmm_cpu": external("cuSPARSE", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu|aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp", "cusparseSpMM", "The hybrid sparse result path prepares sparse metadata and delegates the dense product to cuSPARSE."),
    "aten_sparse_addmv_bsr_cpu": external("cuSPARSE", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp", "bsrmv|cusparseSpMV", "The compressed sparse CUDA addmv implementation calls cuSPARSE."),
    "aten_sparse_addmv_csr_cpu": external("cuSPARSE", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp", "cusparseSpMV", "The CSR CUDA addmv implementation calls cuSPARSE SpMV."),
    "aten_sparse_csr_addmm_cpu": external("cuSPARSE", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp", "cusparseSpMM", "The CSR CUDA addmm implementation calls cuSPARSE SpMM for supported layouts."),
    "aten_sampled_addmm_sparse_csr_cpu": external("cuSPARSE", "LINKED_FIXED_LIBRARY", "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp", "cusparseSDDMM", "The SparseCsrCUDA implementation delegates sampled dense-dense multiplication to cuSPARSE SDDMM."),

    # Header/template libraries compiled into ATen kernels.
    "aten_cumsum": external("CUB", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/ScanUtils.cuh", "cuda::cub::inclusive_scan", "The contiguous innermost-dimension scan uses CUB; other layouts use ATen scan kernels."),
    "aten_cumprod_cpu": external("CUB", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/ScanUtils.cuh", "cuda::cub::inclusive_scan", "The contiguous innermost-dimension scan uses CUB; other layouts use ATen scan kernels."),
    "aten_cumprod_backward_cpu": external("CUB", "ATEN_COMPOSITION", "aten/src/ATen/native/ReduceOps.cpp|aten/src/ATen/native/cuda/ScanUtils.cuh", "reversed_cumsum|cuda::cub::inclusive_scan", "The composite backward formula invokes CUDA cumsum/cumprod scans, which can select CUB for contiguous innermost dimensions."),
    "aten_logcumsumexp_cpu": external("CUB", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/ScanUtils.cuh", "cuda::cub::inclusive_scan", "The scan framework can use CUB for a contiguous innermost dimension."),
    "aten_segment_reduce_lengths_cpu": external("CUB", "CONDITIONAL_TEMPLATE_BACKEND", "aten/src/ATen/native/cuda/SegmentReduce.cu", "cub::DeviceSegmentedReduce::Reduce", "The supported contiguous segment-reduction route uses CUB; other cases use ATen CUDA kernels."),
    "aten_nonzero_out_cpu": external("CUB", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/Nonzero.cu", "cub::DeviceReduce::Sum|cub::DeviceSelect::Flagged", "ATen kernels form flags/indices around CUB selection and reduction primitives."),
    "aten_sort_cpu": external("CUB|Thrust", "CONDITIONAL_TEMPLATE_BACKEND", "aten/src/ATen/native/cuda/SortStable.cu|aten/src/ATen/native/cuda/SortImpl.cu", "cuda::cub::radix_sort_*|thrust::stable_sort_by_key", "The chosen CUDA sorting path depends on dtype, dimension length, stability, and build configuration."),
    "aten_topk_cpu": external("CUB", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/TensorTopK.cu", "cuda::cub::inclusive_sum_by_key", "Large top-k uses CUB scan primitives inside a larger ATen selection algorithm."),
    "aten_randperm_cpu": external("CUB", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/Randperm.cu", "cuda::cub::radix_sort_pairs", "ATen generates random keys and uses CUB radix sort to form the permutation."),
    "aten_masked_scatter_cpu": external("CUB", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/IndexKernel.cu", "cuda::cub::mask_exclusive_sum", "CUB computes mask offsets; ATen kernels validate and scatter values."),
    "aten_masked_select_cpu": external("CUB", "ATEN_COMPOSITION", "aten/src/ATen/native/cuda/IndexKernel.cpp|aten/src/ATen/native/cuda/Nonzero.cu", "index_out|cub::DeviceSelect::Flagged", "masked_select delegates mask indexing to the CUDA indexing/nonzero machinery."),
    "aten_masked_select_serial_cpu": external("CUB", "NO_EXACT_CUDA_COUNTERPART", "aten/src/ATen/native/cuda/IndexKernel.cpp|aten/src/ATen/native/cuda/Nonzero.cu", "masked_select_cuda", "CUDA has one masked-select route rather than the CPU serial helper; that route reaches CUB-backed indexing."),
    "aten_index_put_impl_cpu": external("CUB", "CONDITIONAL_TEMPLATE_BACKEND", "aten/src/ATen/native/cuda/Indexing.cu", "cuda::cub::radix_sort_pairs", "The deterministic/sorted CUDA index_put path uses CUB; the ordinary atomic path does not."),
    "aten_unique_bool_cpu": external("CUB|Thrust", "NO_EXACT_CUDA_COUNTERPART", "aten/src/ATen/native/cuda/UniqueCub.cu|aten/src/ATen/native/cuda/Unique.cu", "run_length_encode|thrust::unique", "CUDA unique selects CUB or Thrust machinery; it has no separate CPU-bool helper counterpart."),
    "aten_unique_consecutive_cpu": external("CUB|Thrust", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/UniqueCub.cu|aten/src/ATen/native/cuda/Unique.cu", "run_length_encode|thrust::unique_by_key", "CUDA unique-consecutive is implemented with template-library scans/run-length operations."),
    "aten_unique_dim_impl_cpu": external("Thrust", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/Unique.cu", "thrust::sort|thrust::unique", "Dimension-wise CUDA unique uses Thrust sorting and uniquing."),
    "aten_unique_dim_template_cpu": external("Thrust", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/Unique.cu", "thrust::sort|thrust::unique", "Dimension-wise CUDA unique uses Thrust sorting and uniquing."),
    "aten_unique_sorted_cpu": external("CUB|Thrust", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/cuda/UniqueCub.cu|aten/src/ATen/native/cuda/Unique.cu", "run_length_encode|thrust::unique", "CUDA unique uses CUB for the common flattened path and Thrust for dimension-wise paths."),
    "aten_mode_cpu": external("Thrust", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/cuda/TensorModeKernel.cu", "thrust::sort_by_key|reduce_by_key", "ATen uses Thrust sorting/reduction/search inside the CUDA mode implementation."),
    "aten_coalesce_sparse_cpu": external("Thrust", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/sparse/cuda/SparseCUDATensor.cu", "thrust::sort_by_key|unique_by_key", "Sparse COO coalescing uses Thrust to sort and group indices, with ATen bookkeeping around it."),
    "aten_sparse_coo_softmax_cpu": external("Thrust", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/sparse/cuda/SoftMax.cu", "thrust::sort|reduce_by_key|exclusive_scan", "The CUDA sparse-softmax path uses Thrust primitives plus ATen kernels."),
    "aten_sparse_coo_softmax_backward_cpu": external("Thrust", "MIXED_EXTERNAL_AND_ATEN_KERNELS", "aten/src/ATen/native/sparse/cuda/SoftMax.cu", "thrust::lower_bound|for_each", "The CUDA sparse-softmax backward path uses Thrust primitives plus ATen kernels."),

    # Template GEMM/attention libraries.
    "aten_nested_bmm_cpu": external("CUTLASS", "TEMPLATE_GENERATED_LIBRARY", "aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu", "cutlass::gemm::device::GemmGrouped", "Nested CUDA bmm instantiates CUTLASS grouped GEMM kernels."),
    "aten_nested_matmul_broadcast_cpu": external("CUTLASS", "ATEN_COMPOSITION", "aten/src/ATen/native/nested/NestedTensorMatmul.cpp|aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu", "bmm_nested_cuda", "The composite nested matmul route reaches CUTLASS grouped GEMM when it lowers to nested bmm."),
    "aten_flash_attention_cpu": external("CUTLASS|cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/transformers/cuda/attention.cu", "_flash_attention_forward|run_cudnn_SDP_fprop", "CUDA scaled-dot-product attention selects among template flash/efficient kernels and cuDNN under runtime guards."),
    "aten_flash_attention_backward_cpu": external("CUTLASS|cuDNN", "CONDITIONAL_BACKEND", "aten/src/ATen/native/transformers/cuda/attention_backward.cu|aten/src/ATen/native/transformers/cuda/attention.cu", "_flash_attention_backward|run_cudnn_SDP_bprop", "CUDA attention backward selects a template or cuDNN backend under runtime guards."),
}


LINKED_PATTERNS = {
    "cuBLAS": re.compile(r"\b(?:cublas\w*|at::cuda::blas::)"),
    "cuDNN": re.compile(r"\b(?:cudnn\w*|raw_cudnn)"),
    "cuSPARSE": re.compile(r"\b(?:cusparse\w*|at::cuda::sparse::)"),
    "cuSOLVER": re.compile(r"\b(?:cusolver\w*|at::cuda::solver::)"),
    "cuFFT": re.compile(r"\bcufft\w*"),
}
TEMPLATE_PATTERNS = {
    "CUB": re.compile(r"\b(?:at::)?cuda::cub::|\bcub::Device"),
    "Thrust": re.compile(r"\bthrust::(?:sort|stable_sort|sort_by_key|stable_sort_by_key|unique|unique_by_key|reduce|reduce_by_key|inclusive_scan|exclusive_scan|scatter|copy|copy_n|find|find_if|count_if|lower_bound|for_each|sequence)\b"),
    "CUTLASS": re.compile(r"\bcutlass::(?:gemm|conv|Status)"),
}


def cuda_dispatch_sources():
    dispatch = defaultdict(set)
    with (CORPUS / "dispatch_kernel_inventory.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for kernel in filter(None, row["fixtures"].split(",")):
                dispatch[kernel].add(row["stub"].split("::")[-1])

    registrations = defaultdict(list)
    pattern = re.compile(r"REGISTER_(?:CUDA_)?DISPATCH\s*\(\s*(?:[\w:]+::)?(\w+)\s*,\s*&?([\w:]+)", re.S)
    native = PYTORCH / "aten/src/ATen/native"
    for path in native.rglob("*"):
        if path.suffix not in {".cu", ".cuh", ".cpp", ".h"} or "/cpu/" in path.as_posix():
            continue
        text = path.read_text(errors="ignore")
        for match in pattern.finditer(text):
            registrations[match.group(1)].append((path, match.group(2)))
    return {
        kernel: [entry for stub in stubs for entry in registrations.get(stub, [])]
        for kernel, stubs in dispatch.items()
    }


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts/correctness"))
    from build_ce_viewer import ATEN_C_PROVENANCE

    native_rows = {}
    with (CORPUS / "native_cuda_audit.csv").open(newline="") as stream:
        native_rows = {row["kernel"]: row for row in csv.DictReader(stream)}
    dispatch = cuda_dispatch_sources()
    rows = []
    for kernel in sorted(ATEN_C_PROVENANCE):
        fixture_source, fixture_token = ATEN_C_PROVENANCE[kernel]
        coarse = native_rows.get(kernel, {}).get("has_native_cuda", "unknown")
        exact = dispatch.get(kernel, [])
        if kernel in OVERRIDES:
            libraries, form, source, symbol, note = OVERRIDES[kernel]
            status = "EXTERNAL_LIBRARY"
            confidence = "HIGH"
        elif exact:
            source = "|".join(sorted({str(path.relative_to(PYTORCH)) for path, _ in exact}))
            symbol = "|".join(sorted({symbol for _, symbol in exact}))
            libraries = ""
            form = "ATEN_CUDA_KERNEL"
            status = "NO_EXTERNAL_LIBRARY_CALL_FOUND"
            confidence = "HIGH"
            note = "The exact CUDA dispatch counterpart is pinned and its implementation contains no computational external-library call. Header-only utility types such as thrust::pair are not counted as library algorithms."
        elif coarse == "no":
            libraries = ""
            form = "NO_DIRECT_CUDA_COUNTERPART"
            source = ""
            symbol = ""
            status = "NO_DIRECT_CUDA_COUNTERPART"
            confidence = "MEDIUM"
            note = "The existing exhaustive CUDA-availability audit found no direct counterpart; a public ATen composition may still execute on CUDA."
        else:
            libraries = ""
            form = "CUDA_PATH_REVIEWED_NO_EXTERNAL_EVIDENCE"
            source = ""
            symbol = ""
            status = "NO_EXTERNAL_LIBRARY_CALL_FOUND"
            confidence = "MEDIUM"
            note = "CUDA support is reported and the reverse audit of cuBLAS/cuDNN/cuSPARSE/cuSOLVER/cuFFT/CUB/Thrust/CUTLASS call sites found no route for this fixture. The exact CUDA symbol is not yet pinned, so this is weaker than a direct dispatch trace."
        rows.append({
            "kernel": kernel,
            "fixture_source": fixture_source,
            "fixture_token": fixture_token,
            "coarse_native_cuda": coarse,
            "audit_status": status,
            "external_libraries": libraries,
            "implementation_form": form,
            "native_cuda_source": source,
            "evidence_symbol": symbol,
            "confidence": confidence,
            "notes": note,
        })

    fields = list(rows[0])
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    external_rows = [row for row in rows if row["audit_status"] == "EXTERNAL_LIBRARY"]
    linked_libraries = {"cuBLAS", "cuDNN", "cuSPARSE", "cuSOLVER", "cuFFT"}
    template_libraries = {"CUB", "Thrust", "CUTLASS"}
    linked_rows = [row for row in external_rows if set(row["external_libraries"].split("|")) & linked_libraries]
    template_rows = [row for row in external_rows if set(row["external_libraries"].split("|")) & template_libraries]
    overturned_no = [row for row in rows if row["coarse_native_cuda"] == "no" and row["audit_status"] != "NO_DIRECT_CUDA_COUNTERPART"]
    counts = defaultdict(int)
    for row in external_rows:
        for library in row["external_libraries"].split("|"):
            counts[library] += 1
    status_counts = defaultdict(int)
    for row in rows:
        status_counts[row["audit_status"]] += 1
    exact_internal = sum(row["implementation_form"] == "ATEN_CUDA_KERNEL" for row in rows)
    reviewed_internal = sum(row["implementation_form"] == "CUDA_PATH_REVIEWED_NO_EXTERNAL_EVIDENCE" for row in rows)
    pytorch_commit = subprocess.check_output(
        ["git", "-C", str(PYTORCH), "rev-parse", "HEAD"], text=True
    ).strip()
    lines = [
        "# Native CUDA external-library calls in the ATen fixture census",
        "",
        f"Pinned PyTorch commit: `{pytorch_commit}`.",
        "",
        "This audit asks what the native PyTorch CUDA path actually calls. It does not reuse the candidate-library classification from `cuda_library_audit.csv`.",
        "",
        f"- Fixtures audited: {len(rows)}",
        f"- External-library paths identified: {len(external_rows)}",
        f"- Paths involving a dynamically linked CUDA library: {len(linked_rows)}",
        f"- Paths involving template-generated CUB/Thrust/CUTLASS code: {len(template_rows)}",
        f"- Exact registered CUDA counterparts with no external call found: {exact_internal}",
        f"- CUDA-supported paths with no external call site found by reverse audit, but without a pinned exact symbol: {reviewed_internal}",
        f"- No direct CUDA counterpart in the existing availability census: {status_counts['NO_DIRECT_CUDA_COUNTERPART']}",
        f"- Earlier `native_cuda_audit.csv` no-CUDA classifications overturned by source evidence: {len(overturned_no)}",
        "",
        "Library counts overlap when one ATen operation has conditional or mixed backends:",
    ]
    lines.extend(f"- {library}: {count}" for library, count in sorted(counts.items()))
    lines.extend([
        "",
        "`CUB`, `Thrust`, and `CUTLASS` denote template-generated device code, not a fixed dynamically linked call. CUDA Runtime allocation/copy calls, compiler math intrinsics, and Thrust utility types such as `thrust::pair` are not counted. No fixture's exact counterpart was found to call cuFFT, cuSOLVER, cuRAND, NCCL, or NPP; helper fixtures adjacent to FFT/factorization code are not the full FFT/factorization operation.",
        "",
        "`CONDITIONAL_BACKEND` does not prove that a benchmark shape selected that backend. Rows marked `NO_EXACT_CUDA_COUNTERPART` or `ATEN_COMPOSITION` describe the public CUDA operation used for comparison, not a one-to-one CUDA version of a CPU-only helper. See the CSV for per-kernel sources, symbols, confidence, and caveats.",
        "",
        "## Identified kernels",
        "",
    ])
    for row in external_rows:
        lines.append(f"- `{row['kernel']}`: {row['external_libraries']} ({row['implementation_form']}); `{row['native_cuda_source']}`; {row['notes']}")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {OUTPUT}")
    print(dict(sorted(status_counts.items())))
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
