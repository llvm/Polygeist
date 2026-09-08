# Publication Orin campaign: 3mm

Native and raised GPU correctness plus one-process 5+5 timing are complete for
canonical LARGE/FP64 inputs.

- Native modified-source PolyBenchGPU: exact 880,000-value pass; device median
  82.369789124 ms (82.349311829–82.401054382, IQR 0.034061432); E2E median
  91.185 ms (91.168–91.238, IQR 0.0555).
- Raised external-library path: exact correctness; compute-device median
  182.088959 ms (181.747070–182.746460, IQR 0.686203); synchronized
  compute-wall median 182.080288 ms; memory-device median 44.718400 ms; E2E
  median 227.371648 ms.
- Device/device ratio: 0.452361x (native is faster).

All builds were cross-compiled on x86. The raised path uses an ABI-only
harness and stored automatically matched IR, not an untouched whole-program
claim. Fixed hardware-state reporting was unavailable, so values remain
publication-pending.
