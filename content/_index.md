---
date: 2017-10-19T15:26:15Z
lastmod: 2019-10-26T15:26:15Z
publishdate: 2018-11-23T15:26:15Z
---

# Polygeist Overview

Polygeist is a new compilation flow that connects the MLIR compiler
infrastructure to cutting edge polyhedral optimization tools. Our goal with
Polygeist is to connect decades of research in the polyhedral model to the new
MLIR compiler infrastructure. 


```c
double foo(double);

double grad_foo(double x) {
    return __enzyme_autodiff(foo, x);
}
```

By differentiating code after optimization, Enzyme is able to create substantially faster derivatives than existing tools that differentiate programs before optimization.

<div style="padding:2em">
<img src="/all_top.png" width="500" align=center>
</div>

## Components

Polygeist is composed of three pieces:

*   A frontend to emit MLIR SCF from a broad range of exisiting C or C++ code.
*   A set of compilation passes to raise SCF constructs to the Affine dialect.
*   A set of compilation passes to have a bi-directional conversion between MLIR and OpenScop exchange format.

## More resources

For more information on Polygeist, please see:

*   The Polygeist [installation guide](/Installation/)

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
