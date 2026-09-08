#!/usr/bin/env python3
"""Generate provenance and legal-comparability metadata for ATen baselines."""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ATEN = HERE.parent

API_BY_CATEGORY = {
    "adaptavg2d": "torch.nn.functional.adaptive_avg_pool2d",
    "adaptavg2d_backward": "torch.ops.aten._adaptive_avg_pool2d_backward",
    "adaptavg3d": "torch.nn.functional.adaptive_avg_pool3d",
    "adaptavg3d_backward": "torch.ops.aten._adaptive_avg_pool3d_backward",
    "adaptmax2d": "torch.nn.functional.adaptive_max_pool2d",
    "adaptmax2d_backward": "torch.ops.aten.adaptive_max_pool2d_backward",
    "add_alpha": "torch.add(alpha=0.75)",
    "add_clamp": "torch.add + torch.clamp",
    "addmm": "torch.addmm",
    "avgpool2d": "torch.nn.functional.avg_pool2d",
    "avgpool2d_backward": "torch.ops.aten.avg_pool2d_backward",
    "avgpool3d": "torch.nn.functional.avg_pool3d",
    "avgpool3d_backward": "torch.ops.aten.avg_pool3d_backward",
    "batchnorm": "torch.nn.functional.batch_norm(training=False)",
    "binary": "torch.{op}",
    "bmm": "torch.bmm",
    "cat": "torch.cat",
    "cat_serial": "torch.cat",
    "cat_sparse_values": "torch.cat(values only)",
    "conv2d": "torch.nn.functional.conv2d",
    "conv3d": "torch.nn.functional.conv3d",
    "convT2d": "torch.nn.functional.conv_transpose2d",
    "cum": "torch.{op}",
    "cumprod_backward": "torch.ops.aten.cumprod_backward",
    "dense_sparse_add": "dense + sparse COO tensor",
    "div_floor": "torch.div(rounding_mode='floor')",
    "dot": "torch.dot",
    "elu_backward": "torch.ops.aten.elu_backward",
    "fractional_maxpool2d": "torch.nn.functional.fractional_max_pool2d",
    "fractional_maxpool3d": "torch.nn.functional.fractional_max_pool3d",
    "gemv": "torch.mv",
    "hardswish_backward": "torch.ops.aten.hardswish_backward",
    "layernorm": "torch.nn.functional.layer_norm",
    "layernorm_backward": "torch.ops.aten.native_layer_norm_backward",
    "layernorm_native": "torch.ops.aten.native_layer_norm",
    "loss": "torch.nn.functional.{op}",
    "loss_bce": "torch.nn.functional.binary_cross_entropy",
    "maxpool2d": "torch.nn.functional.max_pool2d",
    "maxpool3d": "torch.nn.functional.max_pool3d",
    "mean": "torch.mean",
    "mish_backward": "torch.ops.aten.mish_backward",
    "mm": "torch.mm",
    "norm": "torch.norm",
    "outer": "torch.outer",
    "pow_scalar": "torch.pow(Tensor, Scalar)",
    "reduce": "torch.{op}",
    "reduce_arg": "torch.{op}(dim=1)",
    "rmsnorm": "torch.nn.functional.rms_norm",
    "sigmoid_backward": "torch.ops.aten.sigmoid_backward",
    "silu_backward": "torch.ops.aten.silu_backward",
    "softmax": "torch.softmax",
    "softmax_backward": "torch.ops.aten._softmax_backward_data",
    "sort": "torch.sort",
    "sparse_coo_mm": "torch.sparse.mm(COO, dense)",
    "sparse_csr_mm": "torch.sparse.mm(CSR, dense)",
    "sparse_sampled_addmm": "torch.sparse.sampled_addmm",
    "sparse_softmax_backward": "torch.ops.aten._sparse_softmax_backward_data",
    "split_copy": "torch.split_copy",
    "sum_backward": "expand + clone (materialized dense output)",
    "tanh_backward": "torch.ops.aten.tanh_backward",
    "topk": "torch.topk",
    "unary": "torch.{op}",
    "unfold2d": "torch.nn.functional.unfold",
    "weight_norm_backward": "torch.ops.aten._weight_norm_interface_backward",
}

EXACT_COMPOSITIONS = {"aten_add_clamp", "aten_nested_sum_backward_cpu"}
DENSE_MATH_PROXIES = {
    "aten_cat_sparse_cpu",
    "aten_nested_softmax_backward_cpu",
    "aten_sparse_add_values_cpu",
    "aten_sparse_csr_reduce_all_cpu",
    "aten_sparse_mul_cpu",
    "aten_sparse_sum_cpu",
}
STAGE_EQUIVALENTS = {
    "aten_batch_norm",
    "aten_batch_norm_transform_cpu",
    "aten_conv2d_columns_cpu",
}

# These links identify the implementation reached by CUDA dispatch.  Keep this
# list deliberately conservative: an empty source means that CUDA execution was
# measured, but we have not pinned one implementation file because dispatch may
# select among several CUDA/cuDNN implementations.
CUDA_SOURCE_BY_CATEGORY = {
    "adaptavg2d": ("aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu",
                   "Tensor adaptive_avg_pool2d_cuda("),
    "adaptavg2d_backward": (
        "aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu",
        "Tensor adaptive_avg_pool2d_backward_cuda("),
    "adaptavg3d": ("aten/src/ATen/native/cuda/AdaptiveAveragePooling3d.cu",
                   "Tensor adaptive_avg_pool3d_cuda("),
    "adaptavg3d_backward": (
        "aten/src/ATen/native/cuda/AdaptiveAveragePooling3d.cu",
        "Tensor adaptive_avg_pool3d_backward_cuda("),
    "avgpool2d": ("aten/src/ATen/native/cuda/AveragePool2d.cu",
                  "avg_pool2d_out_cuda"),
    "avgpool3d": ("aten/src/ATen/native/cuda/AveragePool3d.cu",
                  "avg_pool3d_out_cuda"),
}


def load_provenance():
    result = {}
    for path in sorted(ATEN.glob("generated*_provenance.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                result[row["kernel"]] = (row["source"], row["token"])
    audit = ATEN / "cuda_library_audit.csv"
    if audit.exists():
        with audit.open(newline="") as stream:
            for row in csv.DictReader(stream):
                result.setdefault(row["kernel"],
                                  (row["source"], row["source_token"]))
    return result


def load_native_fixture_adjudication():
    path = HERE / "native_fixture_adjudication.csv"
    rows = {row["kernel"]: row for row in csv.DictReader(path.open())}
    if not rows:
        raise RuntimeError(f"empty native-fixture adjudication: {path}")
    return rows


def main():
    specs = json.loads((HERE / "resident_shape_specs.json").read_text())
    provenance = load_provenance()
    fixture_adjudication = load_native_fixture_adjudication()
    expected_fixture_rows = {
        spec["kernel"] for spec in specs
        if spec.get("comparison_scope")
        == "REQUIRES_EXPLICIT_NATIVE_FIXTURE_ADAPTER"
    }
    if set(fixture_adjudication) != expected_fixture_rows:
        missing = sorted(expected_fixture_rows - set(fixture_adjudication))
        stale = sorted(set(fixture_adjudication) - expected_fixture_rows)
        raise RuntimeError(
            "native-fixture adjudication is not exhaustive: "
            f"missing={missing}, stale={stale}")
    invalid_legal = sorted(
        kernel for kernel, row in fixture_adjudication.items()
        if row.get("legal_ratio") not in {"yes", "no"})
    if invalid_legal:
        raise RuntimeError(
            f"invalid native-fixture legal_ratio values: {invalid_legal}")
    gpu = {row["kernel"]: row for row in csv.DictReader(
        (HERE / "torch_aten_resident_sync_wall.csv").open())}
    cpu = {row["kernel"]: row for row in csv.DictReader(
        (HERE / "torch_aten_cpu_sync_wall.csv").open())}
    cpu_orin = {row["kernel"]: row for row in csv.DictReader(
        (HERE / "torch_aten_orin_cpu_sync_wall.csv").open())}
    rows = []
    for spec in specs:
        kernel = spec["kernel"]
        source, token = provenance.get(kernel, ("", ""))
        api = API_BY_CATEGORY.get(spec["cat"], "torch.{op}").format(
            op=spec["op"])
        cuda_source, cuda_token = CUDA_SOURCE_BY_CATEGORY.get(
            spec["cat"], ("", ""))
        if kernel in DENSE_MATH_PROXIES:
            comparability = "INTERNAL_STAGE_OR_DENSE_MATH_PROXY"
            legal_ratio = "no"
            note = ("times equivalent dense/value arithmetic, not full sparse "
                    "or nested-tensor dispatch/storage semantics")
        elif kernel in STAGE_EQUIVALENTS:
            comparability = "EXACT_EXTRACTED_STAGE"
            legal_ratio = "yes_stage_only"
            note = "same extracted stage; not a claim about the complete parent operator"
        elif kernel in EXACT_COMPOSITIONS:
            comparability = "EXACT_MATERIALIZED_COMPOSITION"
            legal_ratio = "yes_composition"
            note = "same dense result and layout, implemented by multiple ATen operations"
        elif spec.get("comparison_scope") == "REQUIRES_EXPLICIT_NATIVE_FIXTURE_ADAPTER":
            adjudication = fixture_adjudication.get(kernel)
            if adjudication is None:
                raise RuntimeError(
                    f"native fixture lacks an explicit adjudication: {kernel}")
            if adjudication.get("fixture_op", "") != (spec.get("op") or ""):
                raise RuntimeError(
                    f"stale native-fixture adjudication for {kernel}: "
                    f"{adjudication.get('fixture_op')!r} != {spec.get('op')!r}")
            comparability = adjudication["comparability"]
            legal_ratio = adjudication["legal_ratio"]
            note = adjudication["semantic_note"]
        else:
            comparability = "EXACT_ATEN_OPERATION"
            legal_ratio = "yes"
            note = "shape, dtype, output materialization, and operation semantics aligned"
        rows.append({
            "kernel": kernel, "upstream_source": source,
            "upstream_token": token, "benchmark_api": api,
            "native_cuda_source": cuda_source,
            "native_cuda_token": cuda_token,
            "native_cuda_provenance": (
                "PINNED_IMPLEMENTATION_SOURCE" if cuda_source
                else "MEASURED_CUDA_DISPATCH_SOURCE_NOT_PINNED"),
            "comparability": comparability, "legal_ratio": legal_ratio,
            "semantic_note": note, "shape": spec["shape"],
            "adjudication_source": (
                "native_fixture_adjudication.csv"
                if spec.get("comparison_scope")
                == "REQUIRES_EXPLICIT_NATIVE_FIXTURE_ADAPTER"
                else "built_in_provenance_rule"),
            "dtype": spec["dtype"],
            "shape_selection": spec.get("shape_selection", ""),
            "shape_selection_note": spec.get("shape_selection_note", ""),
            "gpu_status": gpu.get(kernel, {}).get("status", "MISSING"),
            "cpu_status": cpu.get(kernel, {}).get("status", "MISSING"),
            "cpu_orin_status": cpu_orin.get(kernel, {}).get("status", "MISSING"),
            "timing_scope": "best_of_20_synchronized_wall; warmup=5",
        })
    output = HERE / "torch_aten_baseline_provenance.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
