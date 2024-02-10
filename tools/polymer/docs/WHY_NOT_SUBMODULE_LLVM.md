# Why not have LLVM as a submodule to Polymer

We used to have a LLVM submodule in Polymer, which kept track of the specific commit of LLVM that Polymer relies on.

However, we're now moving Polymer as a project within Polygeist, doing so is practically not possible, due to the cyclic dependencies that might be introduced.

Let's say, you have three git repos:

* LLVM-1: the parent of Polymer, which keeps Polymer as its submodule.
* POLYMER
* LLVM-2: the llvm project as a submodule of Polymer.

The directory structure would look like:

```
<LLVM-1>
    |
    +----> <POLYMER>
               |
               +-----> <LLVM-2>
```

In another word, POLYMER is a submodule of LLVM-1, and LLVM-2 is a submodule
of POLYMER.

Suppose the versions of these repos are:

* LLVM-1 is llvm@v1
* POLYMER is polymer@v1
* LLVM-2 is llvm@v0

LLVM-1 and LLVM-2 are of two different versions of the same repo llvm, and normally LLVM-2 is older than LLVM-1.

When you try to make LLVM-2 of the same version as LLVM-1, i.e., update LLVM-2 from llvm@v0 to llvm@v1, the version of POLYMER is changed as well, from polymer@v1 to polymer@v2, and consequently, LLVM-1 will be updated to llvm@v2.
It unfortunately says that LLVM-2 can never catch up with LLVM-1.
