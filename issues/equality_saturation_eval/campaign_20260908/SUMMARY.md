# Equality-saturation ablation

Inputs: 687; runs: 6870; successful: 6865.

## Match coverage

Coverage below uses only inputs with at least one successful run in both arms; whole-input timeouts are reported separately.

| Suite | Inputs | Common-success inputs | Egglog matches | Syntactic matches | Egglog bodies | Syntactic bodies | Egglog-only | Syntactic-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aten | 598 | 598 | 363 | 363 | 417 | 416 | 1 | 1 |
| llama2c | 3 | 3 | 3 | 3 | 6 | 6 | 0 | 0 |
| llama_forward | 20 | 20 | 52 | 52 | 65 | 65 | 0 | 0 |
| mfem_application | 16 | 15 | 250 | 250 | 498 | 498 | 0 | 0 |
| mfem_kernel | 20 | 20 | 128 | 128 | 256 | 256 | 0 | 0 |
| polybench | 30 | 30 | 27 | 24 | 32 | 28 | 4 | 1 |

## Performance

| Metric | Egglog median [Q1, Q3] | Syntactic median [Q1, Q3] | Paired Egglog - syntactic median [Q1, Q3] |
|---|---:|---:|---:|
| Fresh-process wall time (ms) | 826.471 [738.335, 1123.018] | 800.118 [722.551, 1003.846] | 20.798 [-12.295, 93.381] |
| Matcher time (ms) | 365.142 [299.227, 561.919] | 331.847 [287.140, 461.968] | 16.827 [-4.306, 74.352] |
| Peak RSS (KiB) | 66570.000 [65520.000, 69668.000] | 65628.000 [64768.000, 65896.000] | 288.000 [-8.000, 4159.000] |

Completed individual Egglog proofs over 10 seconds: 0.

## Parameters

```json
{
  "repetitions_per_mode": 5,
  "modes": [
    "egglog",
    "syntactic"
  ],
  "parallel_jobs": 1,
  "egglog_iteration_limit": 8,
  "candidate_time_limit_s": 10,
  "candidate_timeout_enforcement": "post-run classification; each fresh input process also has an outer watchdog",
  "distributivity": false,
  "handwritten_semantic_fallback": false,
  "input_ast_node_ceiling": 32,
  "binding_proposal_limit": 8,
  "explicit_egraph_size_limit": null,
  "explicit_memory_limit": null,
  "outer_input_watchdog_s": 120.0,
  "cpu_affinity": 0,
  "python": "/usr/bin/python3",
  "egglog_version": "11.4.0",
  "collect_egraph_sizes": false
}
```

## Egglog-only examples

- `aten/aten_addmm` bodies `[0, 1]` -> `cublasDgemm`
- `polybench/2mm` bodies `[1]` -> `cublasDgemm_alpha_only`
- `polybench/gemm` bodies `[0, 1]` -> `cublasDgemm`
- `polybench/gemver` bodies `[1]` -> `cublasDgemv_alpha`
- `polybench/gemver` bodies `[3]` -> `cublasDgemv_alpha`

## Failures

```json
[
  {
    "suite": "mfem_application",
    "input_id": "mfem_app_navier_tgv_pa_operators_3d",
    "mode": "egglog",
    "repetition": "1",
    "status": "input_timeout"
  },
  {
    "suite": "mfem_application",
    "input_id": "mfem_app_navier_tgv_pa_operators_3d",
    "mode": "egglog",
    "repetition": "2",
    "status": "input_timeout"
  },
  {
    "suite": "mfem_application",
    "input_id": "mfem_app_navier_tgv_pa_operators_3d",
    "mode": "egglog",
    "repetition": "3",
    "status": "input_timeout"
  },
  {
    "suite": "mfem_application",
    "input_id": "mfem_app_navier_tgv_pa_operators_3d",
    "mode": "egglog",
    "repetition": "4",
    "status": "input_timeout"
  },
  {
    "suite": "mfem_application",
    "input_id": "mfem_app_navier_tgv_pa_operators_3d",
    "mode": "egglog",
    "repetition": "5",
    "status": "input_timeout"
  }
]
```
