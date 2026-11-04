# cgeist on the real Vitis HLS headers

cgeist can compile HLS C++ against the headers of a Vitis HLS install
(`<Vitis>/common/technology/autopilot`, the tree the Vitis synthesis front end
reads) instead of stand-ins. Everything is opt-in; without the options below
cgeist's output is unchanged.

## Command line

The Vitis front end's own flags (`db/autopilot.flow.log` of any Vitis run):

```
-I <autopilot> -D__SYNTHESIS__ -D__VITIS_HLS__ -DAESL_SYN -D__HLS_SYN__ -D__DSP48E1__
-include <autopilot>/etc/autopilot_ssdm_op.h -std=gnu++14        (C: -std=gnu99)
```

plus cgeist's Xilinx extensions:

| Option | What |
|--------|------|
| `-Xclang -fxlx-bitwidth` | `int/unsigned __attribute__((bitwidth(N)))` (the headers' `ssdm_int`) is `_BitInt(N)`. Needs the patched clang (`patches/llvm-project/0001-xlx-bitwidth.patch`). |
| `-Xclang -fxlx-apint-promote` | a `_BitInt` no wider than `int` takes part in the integer promotions, as in Vitis' front end. |
| `--xlx-prelude` | force-includes declarations of the Xilinx bit builtins (`include/xlx/xlx_prelude.h`, compiled into cgeist); their calls are lowered to `arith` / `math` ops (`Lib/CGXlxBuiltin.cc`, ported from hls-llvm-project's `CGFPGABuiltin.cpp`). The lowering is active under `--xlx-prelude` or `--xlx-headers` only; without them a function of such a name is an ordinary call. |
| `--xlx-headers` | code generation the headers need: the proxy objects `ap_range_ref`, `ap_bit_ref`, `ap_concat_ref`, `af_range_ref`, `af_bit_ref` are inlined and scalarised, also when they are read inside the `if` / loop regions of the inlined helpers and when one proxy holds references to others (a concatenation of range or bit references), and so are the type-punning unions (`doubleToRawBits`) (`Lib/XlxProxyScalarize.cc`); a value constructed from a string literal (`ap_ufixed<25,11> t[] = {"0x1.a614p0", ...}`, the `_ssdm_string2bits` the Vitis front end folds) is a compile-time constant and such a static table a dense global (`Lib/XlxStringBits.cc`, the Vitis C simulation's conversion: prefixes, `e` / `p` exponents, all quantisation and overflow modes); struct fields of different MLIR types are kept apart; `ns::table[i]` is an ordinary global; a store that the next memory access in the same block overwrites is deleted, such as the guard `if (!op) Base::V = 0;` of the `ap_fixed` conversion (`etc/ap_fixed_base.h:691`), which Vitis' LLVM pipeline removes too (`Lib/XlxDeadStore.cc`); unused `linkonce_odr` functions are dropped. A proxy that still cannot be scalarised (e.g. one passed to a function that is not inlined) is an error (exit code 13), since a memref of a struct holding a memref is not valid MLIR. |
| `-Xclang -fnative-half-type -Xclang -fnative-half-arguments-and-returns` | `half` is `__fp16` with native arithmetic, arguments and returns (Vitis' `-fhls`). |
| `-Xclang -Wno-unknown-attributes` | silences `no_ctor`, `hls_preserve` (streams and the FFT IP only). |
| `--scalar-wrapper-abi` | `ap_int` / `ap_fixed` objects are their `iN`. |

`-Xclang <arg>` passes any argument to the clang frontend (cc1).

Under `--xlx-headers` the pragma parser also accepts the forms the headers use:
a namespace-qualified `variable=ns::name`, and `variable=this` / `*this`
(reported and ignored: it names the object a member function runs on). Without
the option both are errors, as before.

Under `-fxlx-bitwidth` a width below 1 is an error ("bit width must be at least
1") and a width above 8191, the ceiling of the open-source Xilinx front end
(hls-llvm-project `Type.h:1545,3131`), a warning; whether Vitis 2025.1 has the
same ceiling is unconfirmed.

`ap_shift_reg` (`ap_shift_reg.h`): `__fpga_shift_register_peek(Array, Addr)` is
a load of `Array[Addr]` and `__fpga_shift_register_shift(DataIn, Array, Enable)`
an `scf.if (Enable)` around a loop moving every element one place up and a
store of DataIn at 0; the depth is the extent of the array the header casts to
a pointer (`Lib/CGXlxBuiltin.cc`; test `realheaders/shift-register-builtins.cpp`).

## Tests

`Test/HLS/xlx/` (no Vitis install needed; `bitwidth.cpp` needs the patched
clang, feature `xlx-bitwidth`) and `Test/HLS/realheaders/` (the install, from
`$VITIS_AUTOPILOT` or `/mnt/a/Vitis/2025.1/...`; UNSUPPORTED without it).

## Not covered

`hls::stream` (`__fpga_fifo_*`), `hls::task`, `ap_float`, the float / double
`hls_math` functions (closed bitcode), the DSP builtins, the FFT IP, and a
string constructor whose argument is not a string literal (it stays a call to
`_ssdm_string2bits`, which per the header "only takes const string and const
radix", etc/ap_fixed_base.h:458). How Vitis' closed front end rounds a decimal string that lies within
2^-(W-I+4) of a rounding boundary is unknown (the C simulation's sticky-bit
approximation and the exact value can differ there).
