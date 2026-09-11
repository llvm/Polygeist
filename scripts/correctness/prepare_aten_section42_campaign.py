#!/usr/bin/env python3
"""Freeze the ATen Section 4.2 cohort into stable, resumable batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "issues/aten_c_kernels/native_cuda_results"
DEFAULT_OUTPUT = RESULTS / "section42_campaign"
WHOLE_COMPARABILITY = {
    "EXACT_ATEN_OPERATION",
    "EXACT_BENCHMARK_DOMAIN",
    "EXACT_MATERIALIZED_COMPOSITION",
}
CPP_WHOLE_KERNELS = {
    "aten_abs", "aten_acos", "aten_acosh", "aten_asin", "aten_asinh",
    "aten_atan", "aten_atanh", "aten_ceil", "aten_cos", "aten_cosh",
    "aten_erf", "aten_erfc", "aten_exp", "aten_exp2", "aten_expm1",
    "aten_hardswish_backward", "aten_mish_backward",
    "aten_floor", "aten_frac", "aten_log", "aten_log10", "aten_log1p",
    "aten_log2", "aten_neg", "aten_reciprocal", "aten_relu",
    "aten_rsqrt", "aten_sigmoid", "aten_sin", "aten_sinh", "aten_sqrt",
    "aten_square", "aten_tan", "aten_tanh", "aten_trunc",
    "aten_add", "aten_addcdiv", "aten_addcmul", "aten_atan2", "aten_div",
    "aten_fmax", "aten_fmin", "aten_fmod", "aten_hypot",
    "aten_lerp_tensor_cpu", "aten_maximum", "aten_minimum", "aten_mul",
    "aten_pow", "aten_pow_tensor_scalar",
    "aten_gelu", "aten_gelu_cpu_exact", "aten_gelu_cpu_tanh",
    "aten_hardsigmoid", "aten_hardswish", "aten_leaky_relu", "aten_mish",
    "aten_silu", "aten_silu_cpu", "aten_softplus",
    "aten_gelu_backward_cpu_exact", "aten_gelu_backward_cpu_tanh",
    "aten_hardsigmoid_backward", "aten_sigmoid_backward",
    "aten_silu_backward", "aten_softplus_backward", "aten_tanh_backward",
    "aten_copy_cpu", "aten_copy_tensor_array_cpu", "aten_zeros_cpu",
    "aten_transpose_copy", "aten_pixel_shuffle",
    "aten_pixel_shuffle_cpu_backend", "aten_pixel_unshuffle_cpu_backend",
    "aten_angle_real", "aten_clamp", "aten_logaddexp", "aten_logaddexp2",
    "aten_mse_elementwise", "aten_nansum_cpu",
    "aten_adaptive_avg_pool2d", "aten_adaptive_avg_pool3d",
    "aten_avg_pool2d", "aten_avg_pool3d",
    "aten_avg_pool2d_backward_cpu", "aten_avg_pool3d_backward_cpu",
    "aten_bmm", "aten_cross", "aten_cross_cpu_backend",
    "aten_cat_serial_cpu", "aten_repeat_compute_cpu",
    "aten_repeat_tensor_shape_cpu",
    "aten_im2col", "aten_max_pool2d",
    "aten_addmm", "aten_mm", "aten_outer", "aten_sum",
    "aten_elu", "aten_elu_backward", "aten_log_sigmoid_backward_cpu",
    "aten_argmax_cpu", "aten_argmin_cpu", "aten_count_nonzero_impl_cpu",
    "aten_cumprod_cpu",
    "aten_int_mm_out_cpu",
    "aten_conv1d", "aten_conv2d", "aten_conv_transpose2d",
    "aten_conv_transpose3d_cpu",
    "aten_nested_sum_backward_cpu",
    "aten_sampled_addmm_sparse_csr_cpu", "aten_sparse_csr_addmm_cpu",
    "aten_hspmm_cpu", "aten_sparse_addmm_cpu",
}
FRAMEWORK_LEVEL_KERNELS = {"aten_hspmm_cpu", "aten_sparse_addmm_cpu"}
SUPPLEMENTAL_EXACT_RECIPE_KERNELS = {
    "aten_conv_tbc_backward_cpu",
    "aten_conv_tbc_cpu",
    "aten_conv_transpose3d_backward_cpu",
    "aten_conv_transpose3d_grad_weight_cpu",
    "aten_dilated_convolution_cpu",
    "aten_sparse_addmv_bsr_cpu",
}
SUPPLEMENTAL_EXACT_CPP_KERNELS = SUPPLEMENTAL_EXACT_RECIPE_KERNELS
EXACT_INPUT_ALIGNED_KERNELS = SUPPLEMENTAL_EXACT_RECIPE_KERNELS | {
    "aten_conv_transpose2d",
    "aten_conv_transpose3d_cpu",
    "aten_nested_bmm_cpu",
    "aten_nested_matmul_broadcast_cpu",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    status_rows = read_csv(RESULTS / "aten_benchmark_status.csv")
    status = {row["kernel"]: row for row in status_rows}
    specs = {
        row["kernel"]: row
        for row in json.loads((RESULTS / "resident_shape_specs.json").read_text())
    }
    exact_rows = read_csv(RESULTS / "native_exact_region_median5.csv")
    exact = {row["kernel"]: row for row in exact_rows}
    aligned_path = RESULTS / "silicon_threeway_median5.csv"
    aligned = set()
    if aligned_path.exists():
        aligned = {
            row["kernel"] for row in read_csv(aligned_path)
            if row.get("comparison_status") == "ALIGNED"
        }

    whole = {
        kernel
        for kernel, row in status.items()
        if row.get("raised_status") == "VERIFIED_RESIDENT"
        and row.get("comparability") in WHOLE_COMPARABILITY
    }
    exact_names = set(exact)
    overlap = whole & exact_names
    base_cohort = whole | exact_names | FRAMEWORK_LEVEL_KERNELS
    cohort = base_cohort | SUPPLEMENTAL_EXACT_RECIPE_KERNELS
    if (len(whole), len(exact_names), len(overlap), len(base_cohort)) != (115, 71, 1, 186):
        raise SystemExit(
            "cohort drift: expected whole=115 exact_region=71 overlap=1 total=186; "
            f"got {len(whole)}, {len(exact_names)}, {len(overlap)}, "
            f"{len(base_cohort)}"
        )
    missing_specs = sorted(cohort - set(specs))
    if missing_specs:
        raise SystemExit("missing resident specs: " + ", ".join(missing_specs))

    completed = set()
    prior_results = args.output / "results.csv"
    if prior_results.exists():
        for row in read_csv(prior_results):
            if row.get("three_way_status") == "COMPLETE":
                completed.add(row["kernel"])
            if row.get("input_alignment") == "VERIFIED":
                aligned.add(row["kernel"])

    rows = []
    ordered_kernels = sorted(base_cohort) + sorted(
        SUPPLEMENTAL_EXACT_RECIPE_KERNELS - base_cohort)
    for index, kernel in enumerate(ordered_kernels):
        spec = specs[kernel]
        in_exact = kernel in exact_names
        if kernel in whole and in_exact:
            basis = "WHOLE_AND_EXACT_REGION"
        elif in_exact:
            basis = "EXACT_EXTRACTED_REGION"
        elif kernel in SUPPLEMENTAL_EXACT_RECIPE_KERNELS:
            basis = "EXACT_EXTRACTED_REGION"
        else:
            basis = "WHOLE_OPERATION_OR_BOUNDED_DOMAIN"
        supplemental = kernel in SUPPLEMENTAL_EXACT_RECIPE_KERNELS
        rows.append({
            "kernel": kernel,
            "batch": (
                str((len(base_cohort) + args.batch_size - 1) //
                    args.batch_size + 1)
                if supplemental and kernel not in base_cohort
                else str(index // args.batch_size + 1)
            ),
            "cohort_basis": basis,
            "comparability": (
                "FRAMEWORK_LEVEL_ATEN_OPERATION"
                if kernel in FRAMEWORK_LEVEL_KERNELS
                else status[kernel]["comparability"]
            ),
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "recipe_fingerprint": spec["recipe_fingerprint"],
            "cpu_runner": (
                "original_c_harness"
                if kernel == "aten_sparse_addmv_bsr_cpu" else
                "exact_region_harness"
                if (in_exact or kernel in SUPPLEMENTAL_EXACT_CPP_KERNELS or
                    kernel in {"aten_conv_transpose2d",
                               "aten_conv_transpose3d_cpu"})
                else "torch_recipe"
            ),
            "native_gpu_runner": (
                "exact_region_harness"
                if in_exact or kernel in SUPPLEMENTAL_EXACT_CPP_KERNELS else
                "aten_cpp_harness" if kernel in CPP_WHOLE_KERNELS else
                "torch_recipe"
            ),
            "raised_gpu_runner": "polygeist_resident",
            "input_alignment": (
                "VERIFIED"
                if kernel in aligned or kernel in EXACT_INPUT_ALIGNED_KERNELS
                else "PENDING"
            ),
            "current_protocol_state": "COMPLETE" if kernel in completed else "PENDING",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    batch_count = max(int(row["batch"]) for row in rows)
    for batch in range(1, batch_count + 1):
        batch_rows = [row for row in rows if int(row["batch"]) == batch]
        with (args.output / f"batch_{batch:02d}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(batch_rows)

    summary = {
        "kernel_count": len(rows),
        "whole_operation_or_domain_count": len(whole),
        "exact_region_count": len(exact_names | SUPPLEMENTAL_EXACT_RECIPE_KERNELS),
        "overlap_count": len(overlap),
        "overlap": sorted(overlap),
        "batch_size": args.batch_size,
        "batch_count": batch_count,
        "completed_three_way_median5": len(cohort & completed),
        "pending_three_way_median5": len(cohort - completed),
        "verified_input_alignment": len(cohort & aligned),
        "pending_input_alignment": len(cohort - aligned),
        "protocol": {
            "processes_per_kernel_implementation": 1,
            "warmups": 5,
            "samples": 5,
            "statistic": "minimum_of_5",
            "gpu_timing": "synchronized resident wall time",
            "cpu_threads": 1,
            "cpu_affinity": "core_0",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"prepared {len(rows)} kernels in {batch_count} batches at {args.output}; "
        f"complete={summary['completed_three_way_median5']} "
        f"pending={summary['pending_three_way_median5']}"
    )


if __name__ == "__main__":
    main()
