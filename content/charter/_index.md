---
title: "Charter"
date: 2021-09-15T16:12:45Z
draft: false
weight: 10
---

# Introduction

The MLIR ecosystem is growing rapidly while the core infrastructure matures. One thing it is missing is a working C and C++ frontend. Certainly, Clang does an amazing job targeting LLVM IR, but there are many more opportunities to consider by targeting MLIR from C++ before going down to LLVM IR. For example, MLIR code generation and optimization flows can be exercised on well-known benchmarks, higher-level abstractions such as tensor algebra available in MLIR can be more easily connected to C inputs and, on the other hand, C and C++ compiler could benefit from representational capability offered by the MLIR framework for supplementary transformations at the language and library level. Polygeist aims at providing the frontend and potentially feeding relevant components and designs back to Clang and MLIR.

# Core functionality
The primary goal of Polygeist is to develop a C and C++ (and other flavors supported by Clang) frontend for MLIR with the LLVM infrastructure, potentially also connecting to other projects. It will be based on the AST produced by Clang and produce a configurable set of MLIR dialects, i.e. IR components. MLIR provides a set of dialects suitable for modeling higher-level abstractions operated on by a compiler: structured control flow constructs such as loops and conditionals (SCF dialect), program fragments suitable for polyhedral optimization similar to those used in Polly (Affine dialect), directive-based parallelism constructs (OpenMP and OpenACC dialects), etc. While these constructs are not specific to C or C++, they are subject to analyses and transformations that may improve performance. Not being specific to a language also helps share common higher-level transformations, such as loop and memory footprint optimization, similarly to what LLVM IR itself allows but with more semantic information available.  Other dialects, more specific to modeling C, C++ and their sub/supersets supported by Clang, can be introduced directly to Polygeist. Dialects or their components that appear general enough to be used across different languages can be contributed back to MLIR, and language extensions can be contributed back to Clang.

# Extensions

## Compiler Abstractions

Multiple parties expressed interest in an IR for C and C++ to support language-specific analyses and optimizations, and some preliminary work on such an IR has been presented. Polygeist can be seen as a stepping stone in this direction that leverages the existing lexical, syntactic and semantic analysis phases from Clang to obtain an AST, traverses it and produces a set of MLIR dialects that together form the C-level IR. Direct replication of the Clang AST in MLIR may be overly complex, duplicate or otherwise undesirable work. Instead Polygeist can leverage MLIR’s capability for dialect composition to only produce new, higher-level abstractions for a relevant subset of language constructs and otherwise fall back to MLIR’s LLVM IR equivalent, trivially translatable to LLVM IR proper. Examples of such constructs include first-class loop operations, with and without polyhedral limitations, SIMT parallel and synchronization constructs, and type system support for ownership/lifetime analysis. This can assess the feasibility and the utility of progressive introduction of higher-level abstractions into a C++ compiler.

The usual question of the compiler understanding (or not) the standard library can be addressed thanks to the modularity of MLIR dialects that lets us separate the modeling for language features from library features while still using them together in a single representation. In the future, it should remain possible to separate the modeling of library features (MLIR dialects) into a separate project if desired.

## Language Extensions and Features

With an additional compiler internal representation, it becomes possible to better support language extensions in the compiler by translating certain language aspects (namely, calls to intrinsic functions and uses of intrinsic types, #pragma directives, and attributes) to dedicated constructs. This applies to e.g., CUDA, OpenCL and SystemC, all of which can have first-class modeling in the corresponding MLIR dialect and co-exist with other higher-level concepts after being extracted from the AST. The dialect separation for language and library features is also relevant in this case. Each language extension can be provided as (a set of) pluggable MLIR dialects that may or may not be produced depending on the configuration. Polygeist may be a home for abstractions that are common across language extensions, such as specific memory classes or synchronization patterns, yet insufficiently general to warrant inclusion into the MLIR tree.

Language extensions that require modifying the syntax or standard semantics of the language beyond what is currently supported by Clang is out of scope for Polygeist even if it can be modeled in its internal representation.

## Additional Targets

Given an MLIR-based representation, it also becomes possible to prototype C and C++ backends that do not necessarily use LLVM IR. For example, SPIR-V can be produced from in-tree MLIR dialects bypassing LLVM IR (this work is complementary to the ongoing work on the SPIR-V target for LLVM IR). Similarly, upper slices of the high-level synthesis abstraction stacks in CIRCT and ScaleHLS can be targeted from the corresponding library-level abstractions such as those in SystemC.

Polygeist can also help provide more precise control and interception capabilities, recognizing higher-level constructs and transforming them into intrinsic function calls rather than programming with intrinsics directly or relying on the LLVM target to introduce them later in the pipeline. At the same time, there is no intention to replace lower level code generation components such as instruction selection and register allocation.

# Relation To Other LLVM Projects
Polygeist uses clang frontend and AST.
Polygeist uses MLIR’s in-tree dialects for representation and MLIR’s framework to define additional representations.
Polygeist can be connected to Polly’s underlying loop optimizer as one of its transformation passes on high level abstractions (i.e., before lowering to LLVM IR).
Polygeist can be connected to higher levels of the CIRCT abstraction stack.
Flang already uses MLIR framework for its internal representation; some high-level abstractions can be shared between Flang and Polygeist and considered for the MLIR tree if general enough.

# Contributing

We need you! The Polygeist community aims to be open and welcoming. If you'd like to participate, you can do so by.

 * Participate in development on [Github](https://github.com/wsmoses/Polygeist)
 * If you're more interested in improving [Polymer](https://github.com/kumasento/polymer), please also reach out by putting up issues/PRs. 

