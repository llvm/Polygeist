# GENUINE ATen-native extraction (real kernels only) — 2026-08-28
Method: lift ATen's actual kernel VERBATIM from third_party/pytorch/aten/src/ATen/native/cuda/*.cu
(real macros + real launch config), compile with nvcc -arch=sm_87, scp to Orin, run.
NO reimplementations. Library-dispatched ops = real cuBLAS/cuDNN/CUB/cuRAND. Pointwise = flag "not obtainable" (TensorIterator codegen, no source kernel).
Fakes quarantined in reimplemented_NOT_aten_DISCARDED.csv.

- [x] adaptive_avg_pool2d (REAL, 457.6us — vs my fake 304us, proving the point)
- [x] avg_pool2d (323.5us), max_pool2d (337.8us) — real __global__ lifted
- [x] upsample_nearest2d (1326us, real — vs my fake 409us)
- [ ] upsample bilinear/bicubic (UpSample*.cu)
- [ ] pooling backward, adaptive_max, 3d variants
- [ ] library-dispatched: real cuBLAS/cuDNN/CUB/cuRAND (keep the ~50 genuine ones)
- [ ] pointwise (~142): FLAG not-obtainable (no liftable source kernel)

LIFTED SO FAR: 16 real kernels (6 flagged needs-scaffolding)

=========================================================
DONE (2026-08-28) — genuine verbatim-lift extraction
- REAL ATen kernels lifted verbatim from source + built nvcc -arch=sm_87 + run on Orin: 16
  adaptive_avg_pool2d, avg_pool2d, max_pool2d, upsample_nearest2d, adaptive_max_pool2d, reflection_pad1d, max_pool3d, im2col, col2im, max_unpool2d, embedding_renorm, searchsorted, multi_margin_loss, multilabel_margin_loss, upsample_nearest1d, upsample_nearest3d
- FLAGGED (not faked): 8
  upsample_bilinear2d [needs-aten-accessor-scaffolding]; replication_pad2d [needs-aten-accessor-scaffolding]; avg_pool3d [needs-aten-accessor-scaffolding]; grid_sampler_2d [needs-aten-tensorinfo-scaffolding]; nll_loss [needs-aten-accessor-scaffolding]; glu [not-obtainable-tensoriterator-codegen]; replication_pad1d [needs-aten-accessor-scaffolding]; fractional_max_pool2d [needs-aten-accessor-scaffolding]
- HTML rewired: build_ce_viewer.py now reads real_aten_silicon.csv ONLY (discredited native_cuda_silicon.csv de-wired); green bar relabeled 'REAL ATen kernel (verbatim lift)'; served HTTP 200.
- Everything else (TensorIterator pointwise, accessor/TensorInfo kernels) = genuinely not obtainable as a standalone lift without full ATen build; NOT faked.
=========================================================
