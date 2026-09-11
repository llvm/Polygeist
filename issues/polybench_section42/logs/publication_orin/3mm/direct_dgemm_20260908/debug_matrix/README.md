# 3mm diagnosis matrix

- Stored semantic IR passes all 880,000 LARGE/FP64 outputs exactly with runtime timing both disabled and enabled.
- A fresh build with diagnostic `-Dstatic=` exposes canonical source-local `kernel_3mm`, finds six launches (three fills and three GEMMs), emits 18 runtime calls, and passes all 880,000 outputs exactly.
- Without that exposure, explicit `cgeist --function=kernel_3mm` emits an empty module, causing the observed zero-match and missing-`kernel_3mm_impl` link failure.

The implemented generic repair verifies explicit cgeist output and, when the requested symbol is absent, retries the same selection with translation-only static-linkage exposure.  The separately compiled harness and computation remain unchanged.  No benchmark-specific matcher rule or computational kernel is required.
