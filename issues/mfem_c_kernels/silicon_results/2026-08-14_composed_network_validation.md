# MFEM composed-network silicon validation — 2026-08-14

## Configuration

- Hardware: NVIDIA Jetson AGX Orin, SM87, MAXN mode
- CUDA: 12.6
- Problem: `NE=1024`, `D1D=4`, `Q1D=5`, f64
- Measurement: 20 warm iterations; median of process runs 2–4 when the
  correctness gate allowed timing
- Build: the stored post-match `composed.mlir` was passed directly to ABI
  lowering with `polygeist_build.sh --semantic-mlir`, so these executables
  test the exact composition artifacts shown in CE

## Results

| Case | Composition | Correctness | Composed time | Valid performance result |
|---|---:|---:|---:|---|
| Mass3D | 5 launches to 1 network | PASS, max error `6.94e-18` | `658.809592 us` | `16.75x` faster than the old raised path; `4.775x` slower than native MFEM |
| ex35 H1 | unsafe 19-to-15 candidate rejected; 19 pairwise launches retained | PASS, max abs `1.39e-17` | `54796.512006 us` | Correct fallback; no composition speedup |
| IntegrateValue3D | unsafe non-injective 2-to-1 candidate rejected; 2 pairwise launches retained | PASS on the established pairwise path | `5101.929605 us` | Correct fallback; no composition speedup |
| Navier TGV PA operators | three unsafe nonterminal candidates rejected; 70 launches retained | PASS, max abs `8.33e-17` | `973031.571181 us` | Correct fallback; no composition speedup |

Only Mass3D currently provides a valid composed-network speedup. H1,
IntegrateValue3D, and Navier now remain on their correct pairwise paths rather
than producing structurally composed but executable-incorrect networks.

## Failure analysis

The generic composition legality proof is not yet strong enough:

1. Every absorbed contraction has an accumulator/init operand. The trace must
   prove that an internal accumulator is zero, or preserve it in the composed
   algebra. Mass3D satisfies the present assumptions, but the general pass does
   not establish them for every candidate.
2. IntegrateValue3D has a non-injective output view. Its physical affine map
   omits logical mode `d4`, so several logical output positions contribute to
   one physical destination. The pairwise contraction ABI handled this through
   output-mode compaction/reduction. The generic network ABI currently exports
   all five logical modes and therefore loses that reduction semantics.
3. H1 composes into output state also produced by an earlier diffusion stage.
   The network replacement must preserve that cross-stage accumulator and the
   current tensor value, rather than use a stale flat-output snapshot.
4. Navier has three nonterminal networks whose outputs feed later contractions.
   The tensor/submap alias and write-back representation must be proven correct
   across these composed regions before they can be executed safely.

The compiler now rejects candidates unless the destination is a direct
ABI-backed tensor with an injective physical view. It also no longer runs
global CSE at the composition boundary, because distinct `tensor.empty`
scratch roots may be simultaneously live. Supporting the rejected cases as
one network still requires a bufferizable/device-resident representation for
live cross-stage accumulators and explicit non-injective output reductions.

## Logs

- `scripts/correctness/logs/mfem_abs_mass_network_final_20260814_20260814_132317.silicon.log`
- `scripts/correctness/logs/mfem_ex35_h1_composed_final_20260814_20260814_143649.silicon.log`
- `scripts/correctness/logs/mfem_integrate_value_3d_composed_final_20260814_20260814_143649.silicon.log`
- `scripts/correctness/logs/mfem_navier_composed_final_20260814_20260814_143759.silicon.log`
- `scripts/correctness/logs/mfem_ex35_h1_safe_fallback_20260814_20260814_152048.silicon.log`
- `scripts/correctness/logs/mfem_navier_safe_fallback_20260814_20260814_152117.silicon.log`
- `scripts/correctness/logs/mfem_integrate_value_3d_validated_20260814_20260814_151905.silicon.log` (negative test that motivated the non-injective legality rule)
