# Equality-saturation ablation

Inputs: 4; runs: 8; successful: 8.

## Match coverage

Coverage below uses only inputs with at least one successful run in both arms; whole-input timeouts are reported separately.

| Suite | Inputs | Common-success inputs | Egglog matches | Syntactic matches | Egglog bodies | Syntactic bodies | Egglog-only | Syntactic-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aten | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 |
| polybench | 3 | 3 | 8 | 5 | 10 | 6 | 4 | 1 |

## Performance

| Metric | Egglog median [Q1, Q3] | Syntactic median [Q1, Q3] | Paired Egglog - syntactic median [Q1, Q3] |
|---|---:|---:|---:|
| Fresh-process wall time (ms) | 1363.487 [1215.377, 1457.514] | 927.379 [836.115, 1032.081] | 409.232 [224.181, 553.638] |
| Matcher time (ms) | 649.560 [471.242, 830.432] | 390.357 [371.128, 413.755] | 247.354 [89.622, 415.320] |
| Peak RSS (KiB) | 70038.000 [69839.000, 71560.000] | 65888.000 [65798.000, 65995.000] | 4150.000 [4041.000, 5565.000] |

Completed individual Egglog proofs over 10 seconds: 0.

## Parameters

```json
{
  "repetitions_per_mode": 1,
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
  "collect_egraph_sizes": true
}
```

## Egglog-only examples

- `aten/aten_addmm` bodies `[0, 1]` -> `cublasDgemm`
- `polybench/2mm` bodies `[1]` -> `cublasDgemm_alpha_only`
- `polybench/gemm` bodies `[0, 1]` -> `cublasDgemm`
- `polybench/gemver` bodies `[1]` -> `cublasDgemv_alpha`
- `polybench/gemver` bodies `[3]` -> `cublasDgemv_alpha`
