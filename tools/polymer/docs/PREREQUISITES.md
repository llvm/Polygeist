# Polymer Prerequisites

## OS

Polymer in general works on Linux. We have tested Debian-based distros (Ubuntu 18.04 and 20.04) and CentOS 7.6.

## Software

The software requirements from Polymer is a union of [what LLVM requires](https://llvm.org/docs/GettingStarted.html#software) and what Pluto demands ([here](https://github.com/bondhugula/pluto), check the _PREREQUISITES_ section).
Here we won't reiterate what are already well explained in these docs (we do recommend you check out the docs on [GCC setup](https://llvm.org/docs/GettingStarted.html#getting-a-modern-host-c-toolchain) if necessary).

There are several crucial points not covered though.

### LibClang

Although you will build the whole LLVM toolchain soon (for Polygeist), which obviously includes libclang, that version is too advanced, and Pluto (to be specific, [PET](https://repo.or.cz/w/pet.git)) requires a version less than **10**.

To get a working libclang with version less than 10 (we will cover how to specify that to Polymer later), we recommend using what the package manager of your system provides.

* On Ubuntu, you can run the following (these line should work on 18.04 and 20.04):

```shell
# Get Clang-9
sudo apt-get install -y clang-9 libclang-9-dev
# Set it as the default version of clang.
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 100
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 100
```

* On CentOS 7, you can get LibClang (3.4) using the following line:

```shell
yum install clang-devel
```

* Manual install LLVM. There are many many ways of doing this, and here we list one approach we find working (you may need sudo privilege, and the default installation directory is `/usr/local`, [more info](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)):

```shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-9.0.1 # Or any other version below 10.
mkdir build
cd build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install 
```

We will talk about how to refer to LibClang installed by any method.
