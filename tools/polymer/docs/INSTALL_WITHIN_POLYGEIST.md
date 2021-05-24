# Install Polymer within Polygeist

Polymer is now a part of [Polygeist](https://github.com/wsmoses/Polygeist), and our future work will be built on that basis.

We have a [handy script](../scripts/build-with-polygeist.sh) that will clone Polygeist with the version given in [polygeist-version.txt](../polygeist-version.txt) to the upper-level directory, and symlink Polymer to its appropriate position in Polygeist (which should be `mlir/tools/polymer`). This will get around with the tedious submodule sync problem.

To run it:

```sh
# At the root directory of Polymer
./scripts/get-polygeist.sh
```

At the end of this script, `check-polymer` will be launched to perform the regression tests.

Also, the `build` directory in the Polygeist project will be symlinked to the root of Polymer. You can access to all the binaries over there.

