#!/bin/bash -x 

export PATH=$HOME/MLIR-GPU/build/bin:$PATH
export PATH=$HOME/polymer/pluto:$PATH

POLYCONFIG="-D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET "

# dirlist=`ls`
dirlist="seidel-2d"
# dirlist="gemm 2mm 3mm atax bicg"
# dirlist="correlation covariance doitgen mvt "
# dirlist="2mm 3mm atax bicg cholesky correlation covariance deriche doitgen durbin fdtd-2d floyd-warshall gemm gemver gesummv"
# dirlist="gramschmidt heat-3d jacobi-1d jacobi-2d lu ludcmp mvt nussinov seidel-2d symm syr2k syrk trisolv trmm"
for dir in $dirlist; do
	if [ -d $dir ]; then
		#echo $dir
		cd $dir
		rm -f $dir.pluto.exe &>/dev/null
		rm -r $dir.pluto.ll &>/dev/null
		rm -r $dir.polymer.ll &>/dev/null
		rm -f $dir.polymer.exe &>/dev/null

		# compile polymer
		mlir-opt -lower-affine -convert-scf-to-std -convert-std-to-llvm -canonicalize $dir.pluto.mlir | mlir-translate -mlir-to-llvmir > $dir.polymer.ll
		clang $dir.polymer.ll -O3 -o $dir.polymer.exe /home/ubuntu/MLIR-GPU/mlir/tools/mlir-clang/Test/polybench/utilities/polybench.c -lm -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET -I /home/ubuntu/MLIR-GPU/clang/lib/Headers/

		# compile with Pluto
		polycc $dir.c 2>&1 >polycc_`date "+%Y%m%d-%H%M%S"`.log
    clang $dir.pluto.c -O3 -o $dir.pluto.ll -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET -S -emit-llvm -I . 
    clang $dir.pluto.ll polybench.c -O3 -o $dir.pluto.exe -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET -lm

		# clang $dir.pluto.c -O3 -o $dir.clang.exe polybench.c -lm -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET -I ./ 
		
		xclang=$dir.clang
		xpolymer=$dir.polymer
		for i in 1 2 3 4 5; do
			t=`taskset -c 1-1 numactl -i all ./$dir.pluto.exe`
			xclang="$xclang:$t"
			z=`taskset -c 1-1 numactl -i all ./$dir.polymer.exe`
			xpolymer="$xpolymer:$z"
		done
		echo $xclang
		echo $xpolymer

		cd ../

	fi
done
