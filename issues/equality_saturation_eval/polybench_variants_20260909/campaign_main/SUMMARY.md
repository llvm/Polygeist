# Equality-saturation ablation

Inputs: 245; runs: 2450; successful: 2450.

## Match coverage

Coverage below uses only inputs with at least one successful run in both arms; whole-input timeouts are reported separately.

| Suite | Inputs | Common-success inputs | Egglog matches | Syntactic matches | Egglog bodies | Syntactic bodies | Egglog-only | Syntactic-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench | 245 | 245 | 275 | 135 | 349 | 173 | 144 | 4 |

## Performance

| Metric | Egglog median [Q1, Q3] | Syntactic median [Q1, Q3] | Paired Egglog - syntactic median [Q1, Q3] |
|---|---:|---:|---:|
| Fresh-process wall time (ms) | 867.870 [808.783, 974.266] | 847.097 [798.630, 920.781] | 18.090 [-12.473, 85.624] |
| Matcher time (ms) | 413.267 [358.498, 508.278] | 396.592 [354.602, 454.371] | 15.018 [-6.639, 68.924] |
| Peak RSS (KiB) | 67356.000 [66580.000, 70532.000] | 66540.000 [66320.000, 66724.000] | 316.000 [4.000, 4116.000] |

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

- `polybench/2mm::add_zero_lhs` bodies `[0]` -> `memset_zero_2D`
- `polybench/2mm::add_zero_lhs` bodies `[1]` -> `cublasDgemm_alpha_only`
- `polybench/2mm::add_zero_lhs` bodies `[2, 3]` -> `cublasDgemm`
- `polybench/2mm::add_zero_rhs` bodies `[0]` -> `memset_zero_2D`
- `polybench/2mm::add_zero_rhs` bodies `[1]` -> `cublasDgemm_alpha_only`
- `polybench/2mm::add_zero_rhs` bodies `[2, 3]` -> `cublasDgemm`
- `polybench/2mm::mul_one_lhs` bodies `[0]` -> `memset_zero_2D`
- `polybench/2mm::mul_one_lhs` bodies `[1]` -> `cublasDgemm_alpha_only`
- `polybench/2mm::mul_one_lhs` bodies `[2, 3]` -> `cublasDgemm`
- `polybench/2mm::mul_one_rhs` bodies `[0]` -> `memset_zero_2D`
