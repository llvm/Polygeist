#!/usr/bin/env python3
"""Generate torch-native benchmark specs at the EXACT shapes the resident
harness uses (driver cfg["dims"]) — the single source of truth so native and
raised-resident are measured at identical shape + dtype (f32). Output:
resident_shape_specs.json for bench_shaped.py. Run with /usr/bin/python3.10."""
import csv, hashlib, importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NATIVE_RECIPE_VERSION = "2026-09-08-v3"
NATIVE_RECIPE_V4_KERNELS = {
    "aten_angle_complex_scalarized", "aten_as_complex_cpu",
    "aten_complex_scalarized", "aten_conj_complex_scalarized",
    "aten_nested_clone_cpu", "aten_nested_squeeze_cpu",
    "aten_polar_scalarized", "aten_sparse_add_values_cpu",
    "aten_sparse_intersection_apply_cpu",
    "aten_sparse_intersection_launch_cpu", "aten_sparse_mul_cpu",
    "aten_sparse_norm_cpu", "aten_unbind_copy_cpu",
}


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gs = _load("gs", "issues/aten_c_kernels/native_cuda_results/gen_shape_specs.py")
drv = _load("drv", "scripts/correctness/aten_pointwise_graph_silicon.py")


def _complete_library_kernels():
    library_path = ROOT / "issues/aten_c_kernels/cuda_library_audit.csv"
    return {
        row["kernel"] for row in csv.DictReader(library_path.open())
        if row.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE"
        and row.get("counts_as_library_reuse") == "yes"
    }


def main():
    complete_library = _complete_library_kernels()
    category_overrides = {
        "aten_adaptive_avg_pool2d_backward_cpu": "adaptavg2d_backward",
        "aten_adaptive_avg_pool3d_backward_cpu": "adaptavg3d_backward",
        "aten_adaptive_max_pool2d_backward_cpu": "adaptmax2d_backward",
        "aten_mean": "mean",
        "aten_outer": "outer",
        "aten_split_copy_cpu": "split_copy",
        "aten_add_clamp": "add_clamp",
        "aten_avg_pool2d_backward_cpu": "avgpool2d_backward",
        "aten_avg_pool3d_backward_cpu": "avgpool3d_backward",
        "aten_cat_serial_cpu": "cat_serial",
        "aten_cat_sparse_cpu": "cat_sparse_values",
        "aten_conv2d_columns_cpu": "unfold2d",
        "aten_im2col": "unfold2d",
        "aten_cumprod_backward_cpu": "cumprod_backward",
        "aten_div_floor": "div_floor",
        "aten_elu_backward": "elu_backward",
        "aten_fractional_max_pool2d_cpu": "fractional_maxpool2d",
        "aten_fractional_max_pool3d_cpu": "fractional_maxpool3d",
        "aten_hardswish_backward": "hardswish_backward",
        "aten_host_softmax_backward_cpu": "softmax_backward",
        "aten_layer_norm_backward_cpu": "layernorm_backward",
        "aten_layer_norm_cpu_backend": "layernorm_native",
        "aten_mish_backward": "mish_backward",
        "aten_nested_softmax_backward_cpu": "softmax_backward",
        "aten_nested_sum_backward_cpu": "sum_backward",
        "aten_pow_tensor_scalar": "pow_scalar",
        "aten_sampled_addmm_sparse_csr_cpu": "sparse_sampled_addmm",
        "aten_sigmoid_backward": "sigmoid_backward",
        "aten_silu_backward": "silu_backward",
        "aten_sparse_add_values_cpu": "add_alpha",
        "aten_sparse_addmm_cpu": "sparse_coo_mm",
        "aten_sparse_coo_softmax_backward_cpu": "sparse_softmax_backward",
        "aten_sparse_csr_reduce_all_cpu": "reduce",
        "aten_tanh_backward": "tanh_backward",
        "aten_weight_norm_backward_cpu": "weight_norm_backward",
    }
    specs = []
    for kernel in drv._matched_kernels():
        cfg = drv._cfg_for(kernel)
        if not cfg:
            continue
        base = gs.matched_base(kernel)
        cat = gs.CAT.get(base)
        # These fixtures expose sparse storage explicitly.  Benchmark the
        # corresponding sparse ATen operation instead of treating the product
        # of every dimension as a dense element count.
        cat = category_overrides.get(kernel, cat)
        if kernel == "aten_sparse_csr_reduce_all_cpu":
            base = "sum"
        if kernel == "aten_dense_sparse_add_cpu":
            cat = "dense_sparse_add"
        elif kernel == "aten_sparse_csr_addmm_cpu":
            cat = "sparse_csr_mm"
        if cat is None:
            if kernel not in complete_library:
                continue
            # Keep every complete library mapping in the resolved ledger, even
            # when PyTorch has no corresponding CUDA benchmark recipe.  Raised
            # correctness coverage must not depend on native-CUDA availability.
            # bench_shaped emits an explicit skip for adapters it does not yet
            # implement instead of silently dropping the raised case.
            cat = "native_fixture"
        dims = {k: int(v) for k, v in cfg["dims"].items()}
        # total element count = product of dims (matches resident data size for
        # pointwise; structured cats use the dims directly in bench_shaped).
        n = 1
        for v in dims.values():
            n *= max(1, v)
        shape = "_".join(f"{k}={v}" for k, v in cfg["dims"].items())
        kinds = {arg[1] for arg in cfg["args"]}
        if "dptr" in kinds:
            dtype = "f64"
        elif kinds and kinds <= {"iptr", "iscalar"}:
            dtype = "i32"
        else:
            dtype = "f32"
        scalar_args = {
            arg[0]: arg[2] for arg in cfg["args"]
            if arg[1] in {"scalar", "dscalar", "iscalar"}
        }
        explicit_shape = kernel in drv.CASES
        spec = {"kernel": kernel, "op": base, "cat": cat,
                      "dims": dims, "n": n, "shape": shape,
                      "dtype": dtype, "scalar_args": scalar_args,
                      "shape_selection": (
                          "EXPLICIT_STRUCTURED_SHAPE" if explicit_shape else
                          "AUTO_SCALE_LARGEST_ARRAY_APPROX_4194304"),
                      "shape_selection_note": (
                          cfg.get("coverage", "structured operator shape")
                          if explicit_shape else
                          "uniformly scale extracted dimensions; preserve ratios; minimum dimension 2"),
                      "comparison_scope": (
                          "WHOLE_OR_EXPLICITLY_ADJUDICATED" if cat != "native_fixture"
                          else "REQUIRES_EXPLICIT_NATIVE_FIXTURE_ADAPTER")}
        recipe_version = ("2026-09-08-v4"
                          if kernel in NATIVE_RECIPE_V4_KERNELS
                          else NATIVE_RECIPE_VERSION)
        fingerprint_payload = {
            "recipe_version": recipe_version,
            "kernel": kernel, "op": base, "cat": cat, "dims": dims,
            "dtype": dtype, "scalar_args": scalar_args,
        }
        spec["recipe_version"] = recipe_version
        spec["recipe_fingerprint"] = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True,
                       separators=(",", ":")).encode()
        ).hexdigest()[:16]
        specs.append(spec)
    resolved = {spec["kernel"] for spec in specs}
    missing_complete = sorted(complete_library - resolved)
    if missing_complete:
        raise RuntimeError(
            "complete native+raised kernels lack resident benchmark specs: "
            + ", ".join(missing_complete)
        )
    out = Path(__file__).with_name("resident_shape_specs.json")
    out.write_text(json.dumps(specs, indent=0))
    print(
        f"wrote {out} with {len(specs)} benchmark specs at resident shapes; "
        f"covered all {len(complete_library)} complete library mappings"
    )


if __name__ == "__main__":
    main()
