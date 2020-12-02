#!/bin/bash

export PATH=$HOME/MLIR-GPU/build/bin:$PATH

CC=mlir-clang
BASE=$(pwd) 

dirList="linear-algebra/blas 
         linear-algebra/kernels 
         linear-algebra/solvers 
         datamining
         stencils
         medley"

for dir in $dirList; do
  if [ -d $dir ]; then
    cd $dir
    #echo $dir
    for subDir in `ls`; do
      cd $subDir
      #echo "compiling in dir -> $(pwd)"
      S="$CC $(pwd)/$subDir.c $BASE/utilities/polybench.c -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE" 
      S+=" -I $BASE/utilities -I $(pwd) -I $HOME/MLIR-GPU/clang/lib/Headers"
      S+=" -emit-llvm "
      # compile
      $S | opt --O3 -S -o $subDir.ll
      # run 5 times
      x=$subDir
      for i in 1 2 3 4 5; do
        t=`lli $subDir.ll`
        x="$x:$t"
      done
      echo $x
      cd ../
    done 
    if [ $dir == 'linear-algebra/blas' ] || 
       [ $dir == 'linear-algebra/kernels' ] ||
       [ $dir == 'linear-algebra/solvers' ]; then
      cd ../..
    else
      cd ../
    fi
  fi
done
