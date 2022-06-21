---
date: 2017-10-19T15:26:15Z
lastmod: 2019-10-26T15:26:15Z
publishdate: 2018-11-23T15:26:15Z
---

# Polygeist Overview

The MLIR ecosystem is increasing but still misses a working C and C++ frontend.
Certainly, Clang does a fantastic job targeting LLVM IR, but there are many
more opportunities to consider by targeting MLIR from C++ before going down to
LLVM IR. For example, we may want to connect C more easily with higher-level
abstractions available in MLIR. In addition, to maintain C and C++ semantics
by, for example, preserving high-level information such as structured control
flow, OpenMP/GPU parallelism, and lowering C or C++ constructs to user-defined
custom operations.

The following shows a simple example where we use Polygesit to enter
the MLIR lowering pipeline and raise the C code to the Affine dialect.


## Components

Polygeist is composed of two pieces:

*   A frontend to emit MLIR SCF from a broad range of exisiting C or C++ code.
*   A set of compilation passes to raise SCF constructs to the Affine dialect.

## More resources

For more information on Polygeist, please see:

*   The Polygeist [installation guide](/Installation/)
*   Polygeist toy example [toy example](/getting_started/Use_Polygeist.md)

## Citing Polygeist

To cite Polygeist, please cite the following:
```
@inproceedings{polygeistPACT,
  title = {Polygeist: Raising C to Polyhedral MLIR},
  author = {Moses, William S. and Chelini, Lorenzo and Zhao, Ruizhe and Zinenko, Oleksandr},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
  numpages = {12},
  location = {Virtual Event},
  series = {PACT '21},
  publisher = {Association for Computing Machinery},
  year = {2021},
  address = {New York, NY, USA},
  keywords = {Polygeist, MLIR, Polyhedral, LLVM, Compiler, C++, Pluto, Polly, OpenScop, Parallel, OpenMP, Affine, Raising, Transformation, Splitting, Automatic-Parallelization, Reduction, Polybench},
}
```
