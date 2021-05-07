#!/usr/bin/env bash

# Update the submodule reference to the commit of the parent LLVM repo.
#
# This will be useful if you're working under
#
#   <LLVM-1>/mlir/tools/polymer 
#
# while you want to build Polymer as an individual project, that has its
# own copy of LLVM, which locates at 
#
#   <LLVM-1>/mlir/tools/polymer/llvm (let's call it LLVM-2).
#
# As you can see, polymer is a submodule of LLVM-1, and LLVM-2 is a submodule
# of polymer.
#
# The purpose of this script is simply to keep LLVM-2 the same as LLVM-1.
#
# But, it is basically an impossible task:
#
# Let's say, LLVM-1 of version <llvm@v1> refers to <polymer@v1>, and <polymer@v1>
# refers to LLVM-2 of version <llvm@v2>. Now we want to make LLVM-2 become <llvm@v1>,
# which unevitably updates <polymer@v1> to <polymer@v2>, and this will update LLVM-1 
# from <llvm@v1> to <llvm@v3>.
