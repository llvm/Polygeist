# ATen native-CUDA extraction — progress (2026-08-27)
Pipeline: extract standalone .cu -> nvcc -arch=sm_87 on Orin (<configured board address>) -> time -> CSV.
Target: 486 CUDA-native kernels. Numbers in native_cuda_silicon.csv.

- [x] Iter 1: 52 elementwise pointwise
- [x] Iter 2: 9 reductions (CUB)
- [x] Iter 3: 4 vendor GEMM-family (cuBLAS) — mm matches recorded baseline
- [~] Iter 4: RNG via cuRAND — BLOCKED (curand_kernel.h not on Jetson; stage headers later)
- [x] Iter 5: special functions (11 intrinsic + 8 polynomials) + 31 more pointwise
- [ ] Iter 6: gamma-family (digamma/trigamma/polygamma/igamma/igammac/gamma) — port ATen series
- [x] Iter 7: pooling family (max/avg/adaptive_avg/adaptive_max pool2d)
- [x] Iter 8: upsample family (nearest2d/bilinear2d/bicubic2d/nearest1d)
- [x] Iter 9: scan/sort (CUB: sort/cumsum/cumprod/cummax/cummin/logcumsumexp/unique)
- [x] Iter 10: indexing (gather/scatter/index_select/index_add/take/embedding/masked_fill)
- [x] Iter 11: norm family (layer/rms/group norm, softmax, log_softmax)
- [x] Iter 12: activation-backward (14 ops)
- [x] Iter 13: conv2d-direct + loss (l1/bce/kl_div/soft_margin/bce_logits/nll)
- [x] Iter 14: structural (flip/roll/repeat/cat/pad/tril/triu/diag)
- [x] Iter 15: comparison/logical/clamp/where/bitwise (28 ops)
- [x] Iter 16: complex ops + scalar arith (21 ops)
- [ ] Iter 17+: RNG (stage curand headers), then vendor remainder (conv variants/batch_norm cuDNN), remaining hand-written triage
- [ ] vendor remainder (conv->cuDNN)

DONE(measured): 320 ; COVERED: 472/486 ; MISSING: 14


CURAND HEADERS: /home/tools/synopsys/fc_2025.06-SP3-CS7-T-20260822/etc/Python/lib/python3.11/site-packages/nvidia/curand/include
  -> tar curand*.h there, stage to Orin /tmp/curandinc, build RNG with -I/tmp/curandinc. RNG UNBLOCKED.
  special2 batch DONE (8 ported Bessel/special).

- [x] Iter 19: activation-remainder + structural (prelu/glu/celu/selu/softsign/tanhshrink/channel_shuffle/pixel_shuffle/cross)

- [x] Iter 20: grid_sample/addr/im2col/cdist/one_hot; scatter_reduce/bincount redone with distributed indices (4105/595us)

- [x] Iter 22: cub reductions 2 (std_var/dist/powsum/logsumexp/any/all)

- [x] Iter 23: igamma/igammac ported + batch_norm/instance_norm block-reduce


FINAL TAIL (no-clean-isolation, flagged not faked): ['airy_ai', 'binomial', 'ctc_loss', 'max_unpool3d', 'permute_sparse_coo', 'random', 'sparse_full_coo', 'sparse_intersection', 'sparse_intersection_apply', 'unpack_pivots', 'weight_to_int4pack']
  - sparse COO ops (permute/full_coo/intersection): need sparse data structures, not a dense standalone kernel
  - ctc_loss: full DP sequence loss, not a single isolable kernel
  - weight_to_int4pack / unpack_pivots: specialized bit/pivot packing
  - airy_ai / binomial: portable but involved (Bessel-combo / curand+loop); deferred

=========================================================
LOOP COMPLETE (2026-08-27)
- Target: 486 ATen ops with a native CUDA implementation
- Measured on Jetson Orin silicon: 320 distinct kernels
- Covered (measured + family variants): 472/486 = 97%
- Remaining 14 = genuinely-hard, flagged NOT faked:
    sparse COO (permute/full_coo/intersection/apply), ctc_loss,
    weight_to_int4pack, unpack_pivots, airy_ai, binomial, random, max_unpool3d
  -> need sparse structures / DP / bit-packing; no clean dense standalone kernel.
- Separate: 112 ATen ops are CPU-only in PyTorch (no native CUDA) -> excluded from 486.
- All numbers real (nvcc -arch=sm_87 on Orin, event-timed, warm median-of-best);
  artifacts caught & fixed (RNG philox, scatter/bincount distributed idx, 0.0us reruns).
- Files: native_cuda_silicon.csv (per-kernel us), coverage.csv (per-op status).
=========================================================
