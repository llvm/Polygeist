# Patches to Polygeist's llvm-project

Polygeist's `llvm-project/` is a checkout of upstream LLVM at
**`26eb4285b56edd8c897642078d91f16ff0fd3472`** (the revision in
`llvm-project/build/include/llvm/Support/VCSRevision.h`), detached, with its
git directory at `.git/modules/Polygeist/modules/llvm-project` of the HLS
repository (the `.git` file in `llvm-project/` points to a path that does not
exist, so use `GIT_DIR=<HLS>/.git/modules/Polygeist/modules/llvm-project`).
The patches here are applied on top of it. The same change is committed there
as branch **`w11-xlx-bitwidth`** (`ac6056d62`, `546cda75c` on `26eb4285b`),
not checked out, so `llvm-project/build` is untouched; the private Release
build of that branch is `/home/peter/w11/llvm-build` (source
`/home/peter/w11/llvm-project`, the same commits), and the W11-integrate cgeist
(`/home/peter/w11/polygeist-build`) links against it.

| Patch | What | Needed by |
|-------|------|-----------|
| `0001-xlx-bitwidth.patch` | clang: the Xilinx HLS `__attribute__((bitwidth(N)))` as `_BitInt(N)` under `-fxlx-bitwidth` (widths below 1 rejected with their own message, widths above 8191, the open-source Xilinx front end's ceiling, warned); the Vitis integer promotion of narrow `_BitInt` under `-fxlx-apint-promote`. Both cc1 flags are off by default; without them clang is unchanged. Squashed from the private branch `w11-real-headers` of `/home/peter/w11/llvm-project` (commits `ac6056d62`, `546cda75c`). | cgeist on the real Vitis HLS headers (track W11: `-Xclang -fxlx-bitwidth`, the hls-est harness's `--shim real`) |

Apply and rebuild (Release is enough for cgeist; the patch touches `Attr.td`,
so most of clang rebuilds):

```sh
cd <llvm-project> && git apply <Polygeist>/patches/llvm-project/0001-xlx-bitwidth.patch
ninja -C <llvm-build>            # then rebuild cgeist against it
```

Origin: re-implemented from hls-llvm-project (github.com/Xilinx/hls-llvm-project,
commit `ee930469`): `clang/include/clang/Basic/Attr.td:554-559`,
`clang/lib/Sema/SemaType.cpp:2295,2320,6867`, `clang/include/clang/AST/Type.h:1545,3131`,
`clang/lib/Sema/SemaExpr.cpp:704`,
`clang/lib/AST/ASTContext.cpp:5553-5612`. Those files are University of
Illinois/NCSA licensed; Xilinx's additions inside them carry only a
copyright line ("(C) Copyright 2016-2022 Xilinx, Inc."), no licence grant.
MEMOS decision #40 (hls-est tree) allows copying Xilinx's own Apache-2.0 files
(SemaXlxAttr.cpp, CGXlxAttr.cpp, ParseXlxPragma.cpp, CGFPGABuiltin.cpp, ...)
and says that Xilinx edits inside NCSA upstream files whose licence is not
stated are re-implemented, not copied; so this code was written anew from
their behaviour.

The tests in the patch use declarations written for them from the attribute's
behaviour; none reproduces the Vitis-installed headers (the first version of
`SemaCXX/xlx-bitwidth-template.cpp`, in Polygeist commit `048ac028`, followed the
shape of the installed `etc/ap_common.h` `ssdm_int`; replaced in the W11 review
fixes).
