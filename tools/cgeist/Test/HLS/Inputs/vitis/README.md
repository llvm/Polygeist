# Vitis HLS header stand-ins for the cgeist lit tests (`%vitis_include`)

These headers are a copy of the hls-est harness's `exactclass` shim:
`tools/hls-est-validate/shim/exactclass/` in the CIRCT tree (`~/circt`, branch
`hls-est-vitis-match`, files as of commit c4249e886, 2026-10-13, which added
the math-core markers to `hls_math.h` and `-DHLSEST_MATH_MARKERS` to
`cgeist_flags`). Edit the shim there and copy the headers here again; do not
edit them here. The shim's own README in that directory says what is modelled
bit-exactly (checked against the Vitis 2025.1 headers by `check_shim.py` and
`check_shim_cgeist.py`) and what is not.

`ap_int<W>`, `ap_uint<W>`, `ap_fixed<...>` and `ap_ufixed<...>` are classes
whose only data member is the raw `_BitInt(W)` / `unsigned _BitInt(W)`, as in
the real headers in `__SYNTHESIS__` mode, and every operation is the integer
arithmetic of the Vitis headers. cgeist lowers such a single-scalar class to
its scalar only with the opt-in `--scalar-wrapper-abi`, so every RUN line that
uses `%vitis_include` also passes `--scalar-wrapper-abi`: an `ap_fixed<16,5>`
is then an `i16` in the MLIR, an array of them a `memref<...xi16>`.
Without the flag these classes are LLVM structs.

`cgeist_flags` is the shim's file of extra cgeist flags (`--scalar-wrapper-abi`,
`-DHLSEST_MATH_MARKERS`), copied too: the hls-est harness reads it when
`run_hlsest.py --shim polygeist` selects this directory, and the HLSFactory
tests (`hlsfactory/lit.local.cfg`, `%vitis_shim_flags`) read it to build the
harness's command. The other lit tests pass `--scalar-wrapper-abi` themselves
and do not define the macro, so `hls_math.h` calls the libm builtins there.

Not provided (compile errors, not silent differences): assignable range / bit
selects and concatenations (`x.range(7, 0) = v`), string constructors, and the
Vitis headers outside `ap_int.h`, `ap_fixed.h`, `ap_axi_sdata.h`, `ap_cint.h`,
`hls_half.h`, `hls_math.h`, `hls_stream.h` and `hls_vector.h`. `hls::stream`
operations are declared, not defined (calls to external functions). The
HLSFactory designs these gaps stop are listed in `hlsfactory/SUMMARY.md`.

These headers replaced the earlier minimal stand-ins (a 64-bit integer behind
every `ap_[u]int<W>` and a `double` behind every `ap_fixed`) on 2026-10-11
(MEMOS decision #10 of the hls-est project; MEMOS decision #9 keeps
`--scalar-wrapper-abi` opt-in).
