#!/bin/bash

export PATH=$HOME/MLIR-GPU/build/bin:$PATH

CC='clang -O3'
LDFLAGS=-lm
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
      S=$CC
      S+=" -I $BASE/utilities -I $(pwd)"
      S+=" $BASE/utilities/polybench.c $subDir.c -DPOLYBENCH_TIME -DPOLYBENCH_NO_FLUSH_CACHE $LDFLAGS -o $subDir.exe"
      # compile
      $S
      # run 5 times
      x=$subDir
      for i in 1 2 3 4 5; do
        t=`./$subDir.exe`
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
