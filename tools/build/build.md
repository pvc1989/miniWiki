# 构建工具
## 手动构建
### 构建过程
对于用静态语言 (例如 C/C++) 编写的程序, 必须经过`构建` [build] 才能得到可以运行的程序.
下面用一个简单的例子来说明构建过程的主要步骤.

#### 源文件
假设有如下简单的 C 语言项目:
```
demo
├── include
│   └── math.h
├── src
│   └── math.c
└── test
    └── test_math.c
```
各文件大致内容如下:
- [`include/math.h`](./demo/include/math.h) --- 声明函数 `factorial()`, 用于计算正整数的阶乘.
- [`src/math.c`](./demo/src/math.c) --- 实现 `factorial()` 的功能.
- [`test/test_math.c`](./demo/test/test_math.c) --- 在 `main()` 中调用 `factorial()`, 测试其正确性.

为叙述方便, 下面用环境变量 `PROJECT_PATH` 表示 `demo` 的完整路径.
为避免污染源文件目录, 应当在一个独立于 `PROJECT_PATH` 的空目录里进行构建.

#### 编译
采用默认的编译选项:
```shell
# 编译 src/math.c, 得到二进制的目标文件 math.o
cc -c ${PROJECT_PATH}/src/math.c
# 编译 test/test_math.c, 得到二进制的目标文件 test_math.o
cc -c ${PROJECT_PATH}/test/test_math.c
```

#### 打包
两种打包方式:
```shell
# 将 math.o 打包为静态库 libmath.a
ar -rcs libmath.a math.o
# 将 math.o 打包为动态库 libmath.so
cc -shared -fpic -o libmath.so math.o 
```

#### 链接
三种链接方式:
```shell
# 将 test_math.o 和目标文件 math.o 链接进 test_math_o
cc -o test_math_o test_math.o math.o
# 将 test_math.o 和动态库 libmath.so 链接进 test_math_so
cc -dynamic -o test_math_so test_math.o -L. -lmath
# 将 test_math.o 和静态库 libmath.a 链接进 test_math_a
cc -static -o test_math_a test_math.o -L. -lmath
```
⚠️ 在 macOS 系统下, [无法创建 statically linked binaries](https://developer.apple.com/library/archive/qa/qa1118/_index.html), 因此第三种方式不被支持.

### 使用构建工具的动机
手动构建的缺点:
- 代码更新后, 重新构建的过程非常繁琐.
- 构建参数 (编译和链接选项) 容易写错.
- 构建参数无法体现在源代码中.

好的自动构建工具应当具有以下特性:
- 代码更新后, 能够自动识别并更新需要重新编译和链接的文件.
- 不依赖于具体环境 (操作系统, 编译器).
- 构建参数作为源代码的一部分, 保存在构建文件中.

## GNU Make

## CMake
### 参考资料
#### 官方文档
- [帮助文档](https://cmake.org/cmake/help/latest/)
  - [cmake(1)](https://cmake.org/cmake/help/latest/manual/cmake.1.html) --- 主程序 (命令行界面)
  - [cmake-buildsystem(7)](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html) --- 系统配置
  - [cmake-commands(7)](https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html) --- 命令列表
  - [cmake-language(7)](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html) --- 语法定义

#### 视频教程
- [Programming in C++](https://www.ece.uvic.ca/~frodo/cppbook/) by Michael Adams from University of Victoria
  - Lecture Slides [(Version 2019-02-04)](https://www.ece.uvic.ca/~frodo/cppbook/downloads/lecture_slides_for_programming_in_c++-2019-02-04.pdf)
  - Video Presentations (YouTube)
    - Build Tools (Make and CMake)
      - [Introduction](https://youtu.be/FPcK_swg-f8)
      - [Make --- Introduction](https://youtu.be/FsGAM2pXP_Y)
      - [CMake --- Introduction](https://youtu.be/Ak6cGZshduY)
      - [CMake --- Examples](https://youtu.be/cDWOECgupDg)

### 术语
- `源文件目录` [source-dir] (又称 `源文件树` [source-tree]) --- 项目源码文件的 (顶层) 目录, 必须含有一个 `CMakeLists.txt` 文件.
- `构建目录` [build-dir] (又称 `构建树` [build-tree], `二进制树` [binary-tree]) --- 存放构建产物 (目标文件, 库文件, 可执行文件等) 的目录.
- `内部构建` [in-source build] --- 构建目录在源文件目录下 (⚠️会污染源文件目录).      
- `外部构建` [out-of-source build] --- 构建目录在源文件目录外 (推荐).                  
- `构建配置` [build-configuration] --- 由一组构建工具 (编译器, 链接器) 选项所构成的构建参数集.

### `cmake` 命令
CMake 参与的构建过程可以分为以下两个阶段:
1. CMake 读取 `CMakeLists.txt` 文件, 生成`本地构建工具` [native-build-tool] (例如 `make`) 所需的`本地构建文件` [native-build-file] (例如 `makefile`):
```shell
cmake [<options>] <source-dir>
cmake [<options>] <existing-build-dir>
cmake [<options>] -S <source-dir> -B <build-dir>
```
2. 本地构建工具读取 (上一步生成的) 本地构建文件, 调用`本地工具链` [native-tool-chain] (预处理器 [preprocessor], 编译器 [compiler], 汇编器 [assembler], 链接器 [linker]) 进行构建.
这一步可以借助于 CMake 以跨平台的方式来完成:
```shell
cmake --build <build-dir> [<options>] [-- <build-tool-options>]
```

#### 常用选项
```shell
# 查看帮助
cmake --help[-<topic>]
# 查看版本号
cmake --version
# 打开项目
cmake --open <dir>
# 将 CMake 变量 var 的值设为 value
cmake [{-D <var>=<value>}...] -P <cmake-script-file>
# 运行外部程序
cmake -E <command> [<options>]
# 查找包
cmake --find-package [<options>]
```

### `CMakeLists.txt` 文件
`CMakeLists.txt` 是驱动 CMake 程序运行的脚本文件, 它由`命令` [command] 和`注释` [comment] 组成:

- 命令的名称`不区分大小写`, 形式上与函数调用类似.
- 命令的操作对象称为`变量` [variable], 变量的名称`区分大小写`.
- 注释一般以 `#` 开始, 至行尾结束.

#### 常用脚本命令
```cmake
# 限制 CMake 的最低版本
cmake_minimum_required(VERSION 3.1)

# 向终端输出信息
message("Hello World")

# 为变量 X 赋值
set(X "Hello World")
# 取变量 X 的值
message(${X})
```
