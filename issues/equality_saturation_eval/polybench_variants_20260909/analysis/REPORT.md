# Controlled PolyBench equality-saturation variants

Thirty raised PolyBench inputs were offered ten independent, name-agnostic algebraic transformations. 245 variants were applicable and all passed MLIR verification; 55 combinations were recorded as not applicable.

The primary nine-transformation campaign contains 245 inputs and 2450 fresh matcher processes. 2450 succeeded.

## Recovery

For the controlled headline, the target set contains only identities both arms found in the unmodified input. Across 221 such variant/identity opportunities, Egglog retained 207 (93.7%), while exact syntax retained 98 (44.3%).

Against the broader unmodified-Egglog target set, Egglog recovered 241/255 (94.5%) and exact syntax recovered 101/255 (39.6%).

| Variant | Inputs | Common opportunities | Egglog | Exact syntax | Egglog-only target recoveries | Matcher delta median ms |
|---|---:|---:|---:|---:|---:|---:|
| add_zero_lhs | 30 | 30 | 27 | 0 | 31 | 10.929 |
| add_zero_rhs | 30 | 30 | 27 | 0 | 31 | 17.766 |
| mul_one_lhs | 30 | 30 | 27 | 0 | 31 | 18.926 |
| mul_one_rhs | 30 | 30 | 27 | 0 | 31 | 12.670 |
| reassociate_add | 6 | 5 | 5 | 4 | 3 | 8.553 |
| reassociate_mul | 6 | 6 | 4 | 4 | 1 | 0.389 |
| swap_add | 26 | 30 | 30 | 30 | 4 | 22.358 |
| swap_add_mul | 30 | 30 | 30 | 30 | 4 | 25.822 |
| swap_mul | 27 | 30 | 30 | 30 | 4 | 17.971 |

## Cost

Across 1225 paired repetitions, the Egglog-minus-syntax matcher delta was 15.018 ms median [-6.639, 68.924]. The peak-RSS delta was 0.309 MiB median.

## E-graph diagnostic

A separate one-repetition diagnostic covered 63 Egglog-benefit variants. The largest proof had 42 nodes and 17 classes; the median per-input largest proof had 18 nodes and 9 classes. No completed proof exceeded 10 seconds.

## Stress case

The composed `identity_pair` mutation applies `(x + 0) * 1` at every floating-point generic yield. It is excluded from primary aggregates after the 2mm Egglog pilot hit the 120-second whole-input watchdog; the retained stress rows document that scaling failure.

## Remaining robustness gaps

The 14 Egglog misses are explicit: four are `cublasDaxpby` composition misses in Gesummv, five are `cublasDsyr2k` misses, and five are `cublasDsyrk` misses. These composition recognizers still impose structural preconditions outside the scalar Egglog equivalence check.

## CPU execution validation

All 10 extracted-kernel CPU pilot variants built and ran with host OpenBLAS/CBLAS, one thread; complete LARGE FP64 output comparisons passed at rtol 0.0005 and atol 0.011. The generated PolyBench harness declares but does not define the selected kernel, so the transformed object is the sole implementation and no symbol replacement is used.

## Interpretation boundary

These are controlled variants of already-raised Linalg IR. They measure matcher robustness and compilation cost on the x86 host, not frontend raising coverage or application runtime. Floating-point reassociation is judged under the study's numerical-equivalence contract rather than bit-for-bit IEEE evaluation order.
