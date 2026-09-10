# Equality-saturation ablation

Inputs: 63; runs: 126; successful: 126.

## Match coverage

Coverage below uses only inputs with at least one successful run in both arms; whole-input timeouts are reported separately.

| Suite | Inputs | Common-success inputs | Egglog matches | Syntactic matches | Egglog bodies | Syntactic bodies | Egglog-only | Syntactic-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench | 63 | 63 | 156 | 19 | 194 | 22 | 140 | 3 |

## Performance

| Metric | Egglog median [Q1, Q3] | Syntactic median [Q1, Q3] | Paired Egglog - syntactic median [Q1, Q3] |
|---|---:|---:|---:|
| Fresh-process wall time (ms) | 1120.883 [967.863, 1448.721] | 947.169 [888.527, 1073.628] | 158.858 [34.803, 338.129] |
| Matcher time (ms) | 665.741 [495.998, 938.147] | 492.985 [417.741, 578.366] | 159.105 [32.897, 343.676] |
| Peak RSS (KiB) | 70884.000 [70604.000, 71248.000] | 66592.000 [66478.000, 66814.000] | 4324.000 [4120.000, 4490.000] |

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
