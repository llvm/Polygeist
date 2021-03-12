user=9001
group=1000
polymer=/workspace

build-docker: test-docker
	docker run -it -v $(shell pwd):/workspace polymer20:latest /bin/bash \
	-c "make build"
	echo "Polymer has been installed successfully!"

test-docker:
	(cd docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) . --tag polymer20)

shell:
	docker run -it -v $(shell pwd):/workspace polymer20:latest /bin/bash

build_:
	set -e # Abort if one of the commands fail
	# build LLVM
	# mkdir -p $(polymer)/llvm/build
	# (cd $(polymer)/llvm/build; \
	#  cmake ../llvm \
	#  -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
	#  -DLLVM_TARGETS_TO_BUILD="host" \
	#  -DLLVM_ENABLE_ASSERTIONS=ON \
	#  -DCMAKE_BUILD_TYPE=DEBUG \
	#  -DLLVM_BUILD_EXAMPLES=OFF \
	#  -DLLVM_ENABLE_RTTI=OFF \
	#  -DLLVM_INSTALL_UTILS=ON \
	#  -DCMAKE_C_COMPILER=clang-9 \
	#  -DCMAKE_CXX_COMPILER=clang++-9 \
	#  -G Ninja || exit 1; \
	#  ninja || exit 1; \
	#  ninja check-mlir || exit 1)

	# build polymer
	mkdir -p $(polymer)/build
	(cd $(polymer)/build; \
	 cmake .. \
	 -DCMAKE_BUILD_TYPE=DEBUG \
	 -DMLIR_DIR=$(polymer)/llvm/build/lib/cmake/mlir \
	 -DLLVM_DIR=$(polymer)/llvm/build/lib/cmake/llvm \
	 -DLLVM_ENABLE_ASSERTIONS=ON \
	 -DCMAKE_C_COMPILER=clang-9 \
	 -DCMAKE_CXX_COMPILER=clang++-9 \
	 -DLLVM_EXTERNAL_LIT=$(polymer)/llvm/build/bin/llvm-lit \
	 -G Ninja || exit 1; \
	 ninja -j4 || exit 1)
	(cd $(polymer)/build; LD_LIBRARY_PATH=$(polymer)/build/pluto/lib:$LD_LIBRARY_PATH ninja check-polymer)

clean: clean_polymer
	rm -rf $(polymer)/llvm/build

clean_polymer:
	rm -rf $(polymer)/build
