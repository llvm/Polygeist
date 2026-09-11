#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="$(cd "$HERE/.." && pwd)"
OUT="$HERE/build/native"
CC="${CC:-clang}"
ITERS="${BENCH_ITERS:-10000}"
mkdir -p "$OUT"

build() {
  "$CC" -O3 -DBENCH_ITERS="$ITERS" "$CORPUS/normalized/$1" "$HERE/$2" \
    -lm -o "$OUT/$3"
}

build diffusion_stage_sliced.c ex1_diffusion_3d.c ex1_diffusion_3d
build curlcurl3_stage_sliced.c ex3_curlcurl_3d.c ex3_curlcurl_3d
build divdiv3_stage_sliced.c ex4_divdiv_3d.c ex4_divdiv_3d
build convection_stage_sliced.c ex9_convection_2d.c ex9_convection_2d
build mass_stage_sliced.c ex9_mass_2d.c ex9_mass_2d

for executable in "$OUT"/*; do "$executable"; done
