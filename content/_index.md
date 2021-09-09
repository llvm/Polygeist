---
date: 2017-10-19T15:26:15Z
lastmod: 2019-10-26T15:26:15Z
publishdate: 2018-11-23T15:26:15Z
---

# Polygeist Overview

TODO


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

Enzyme is composed of four pieces:

*   An optional preprocessing phase which performs minor transformations that tend to be helpful for AD.
*   A new interprocedural type analysis that deduces the underlying types of memory locations
*   An activity analaysis that determines what instructions or values can impact the derivative computation (common in existing AD systems).
*   An optimization pass which creates any required derivative functions, replacing calls to `__enzyme_autodiff` with the generated functions.

## More resources

For more information on Enzyme, please see:

*   The Enzyme [getting started guide](/getting_started/)
*   The Enzyme [mailing list](https://groups.google.com/d/forum/enzyme-dev) for any questions.
*   Previous [talks](/talks/).

## Citing Enzyme

To cite Enzyme, please cite the following:
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
