---
title: Doxygen
---

# Build

定义一个安装路径：

```shell
MY_INSTALL=<path-to-install>
```

构建并安装 Clang：

```shell
mkdir clang && cd clang
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.1/llvm-project-17.0.1.src.tar.xz
tar -xvf llvm-project-17.0.1.src.tar.xz
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -G Ninja -S ../llvm-project-17.0.1.src/llvm -B .
cmake --build .
cmake --install . --prefix $MY_INSTALL
```

构建并安装 Doxygen：

```shell
mkdir doxygen && cd doxygen
wget https://www.doxygen.nl/files/doxygen-1.9.8.src.tar.gz
tar -xvf doxygen-1.9.8.src.tar.gz
mkdir build && cd build
cmake -Duse_libclang=ON -DClang_DIR:PATH=${MY_INSTALL}/lib/cmake/clang -S .. -B .
cmake --build .
cmake --install . --prefix $MY_INSTALL
```

