---
title: 批量构建
---

# 手动构建
## 手动构建过程
用静态语言（例如 [C++](../cpp.md)）编写的程序，必须经过**构建 (build)** 才能得到**可运行的 (runnable)** 软件。
《[链接](../../csapp/7_linking.md)》介绍了构建所涉及的*目标文件*、*静态库*、*静态链接*、*共享库*、*动态链接*等概念。

下面用一个简单的例子来说明构建的主要步骤。

### 源文件 (Source Files)
假设有如下简单的 C 语言项目：
```shell
.
├── include
│   └── math.h  # 声明函数 `factorial`，用于计算正整数的阶乘。
├── src
│   └── math.c  # 实现 `factorial` 的功能。
└── test
    └── math.c  # 测试 `factorial` 的功能。
```
创建环境变量：
- 为叙述方便，下面用环境变量 `SOURCE_DIR` 表示源文件根目录 `./` 的完整路径。
- 为避免污染 `$SOURCE_DIR`，应当在一个（用环境变量 `BUILD_DIR` 表示的）空目录里构建。

```shell
SOURCE_DIR=$(pwd)
mkdir _build
BUILD_DIR=$SOURCE_DIR/_build
```

### 编译 (Compile)

```shell
cd ${BUILD_DIR}
# 将 源文件 src/math.c 编译为 目标文件 src_math[.pic].o
cc -o src_math.o           -c ${SOURCE_DIR}/src/math.c
cc -o src_math.pic.o -fpic -c ${SOURCE_DIR}/src/math.c
# 将 源文件 test/math.c 编译为 目标文件 test_math.o
cc -I${SOURCE_DIR}/include -o test_math.o -c ${SOURCE_DIR}/test/math.c
```

其中 `-fpic` 表示生成**[位置无关代码 (position-independent code)](../../csapp/7_linking.md#pic)**。

### 打包 (Package)

```shell
cd ${BUILD_DIR}
# 将 目标文件 src_math.o     打包为 静态库 libmath.a
ar -rcs libmath.a src_math.o
# 将 目标文件 src_math.pic.o 打包为 动态库 libmath.so
cc -shared -o libmath.so src_math.pic.o
```

### 链接 (Link)

```shell
cd ${BUILD_DIR}
# 将 目标文件 test_math.o 及 src_math.o 链接进可执行文件 test_math_o
cc -o test_math_o test_math.o src_math.o
# 将 目标文件 test_math.o 及 动态库 libmath.so 链接进 test_math_so
cc -o test_math_so test_math.o -Wl,-rpath,${BUILD_DIR} -L${BUILD_DIR} -lmath
# 将 目标文件 test_math.o 及静态库 libmath.a 链接进 test_math_a
cc -static -o test_math_a test_math.o -L${BUILD_DIR} -lmath
```

⚠️ [在 macOS 下，无法创建 statically linked binaries](https://developer.apple.com/library/archive/qa/qa1118/_index.html)，因此无法实现第三种方式。

### 运行 (Run)
```shell
cd ${BUILD_DIR}
./test_math_o
./test_math_so
./test_math_a
```
运行结果均为：
```
factorial(0) == 1
factorial(1) == 1
factorial(2) == 2
factorial(3) == 6
factorial(19) == 121645100408832000
factorial(20) == 2432902008176640000
factorial(21) == -4249290049419214848 (overflowed)
factorial(20) / factorial(19) == 20
factorial(21) / factorial(20) == -1 (overflowed)
```
其中 `factorial(21)` 的值超出了 `long` 可容纳的范围，发生**上溢 (overflow)**。

### 清理 (Clean)
```shell
cd ${BUILD_DIR}
rm -rf *
```

## 使用构建工具的动机

|                |            手动构建            |        （理想的）自动构建        |
| :------------: | :----------------------------: | :------------------------------: |
| 更新*源代码*后 |       重新构建的过程繁琐       | *自动识别*并更新需要受影响的文件 |
| 编译及链接选项 | 依赖于环境（操作系统、编译器） |           不依赖于环境           |
|    （同上）    |       无法体现在源代码中       |         是源代码的一部分         |

# GNU Make<a href id="GNU-Make"></a>
## 参考资料
- [官方文档](https://www.gnu.org/software/make)

## `make` 命令<a href id="make-cmd"></a>
一般形式：
```shell
make [options] [targets]
```
其中，`options` 表示一个或多个**选项 (option)**，`targets` 表示一个或多个**目标 (target)**，实际使用时不写 `[]`。

### 选项
常用选项：

|     选项      |                   含义                   |
| :-----------: | :--------------------------------------: |
|     `-n`      |  显示（但不实际执行）将要执行的构建命令  |
| `-f filename` | 用名为 `filename` 的文件驱动 `make` 程序 |
|     `-k`      |   即使部分目标失败，仍继续构建其他目标   |

### 目标
一个**目标 (target)** 表示一个定义在 [`Makefile`](#Makefile) 中的构建任务，通常为**可执行文件(executable file)** 或**库 (library)** 的文件名，也可以只是一个**标签 (tag)**。
如果没有为 `make` 指定目标，则以 `Makefile` 中的第一个目标为默认目标。

一个目标可以被重复构建多次。
每次构建前，`make` 会自动检查该目标的**依赖项 (prerequisite)**。只有依赖项需要被更新时，才会在依赖项全部被更新后，重新构建该目标。
这项检查是递归的，因此最终将传递到被更新过的源文件上。

## `Makefile` 文件<a href id="Makefile"></a>
`Makefile` 是驱动 [`make` 命令](#make-cmd)的**脚本 (script)**：

- 默认文件名为 `Makefile` 或 `makefile`。
- 也可以用其他文件名，但必须在 `make` 后面用 `-f filename` 来指定。

`Makefile` 主要用来定义构建*目标*，一般形式为：

```Makefile
# comments
targets : prerequisites
	commands
```
各字段的含义如下：

|      字段        |                            含义                         |
| :-------------: | :-----------------------------------------------------: |
|    `targets`    |                  一个或多个*目标*                  |
| `prerequisites` | 当前 `targets` 的依赖项，一般是文件名，也可以是其他 `targets` |
|   `commands`    | 编译、链接、系统命令，缩进必须用制表符；每一行都是独立进程    |
|    `comment`    |                注释，以 `#` 开始，到行尾结束               |

### 目标
一般情况下，一个目标对应于一个同名文件，构建该目标就是构建该文件。

除此之外，有一类特殊的目标，只表示一组构建行为，而不生成对应的同名文件。常用的有 `all` 和 `clean`。
这类目标统一地被标注为 `.PHONY` 这个特殊目标的依赖项：

```Makefile
.PHONY: all clean
```
虽然 `all` 和 `clean` 在**语法 (syntax)** 上没有特殊含义，但几乎所有项目都是按如下**语义 (semantics)** 来使用的：
- `all` 用于构建所有当前 `Makefile` 中的所有目标。
- `clean` 用于删除构建过程中生成的所有目标文件和可执行文件。

### 变量
常用的内置变量：
```Makefile
CC        # C 编译命令
CFLAGS    # C 编译选项
CXX       # C++ 编译命令
CXXFLAGS  # C++ 编译选项
ARCLAGS   # 打包选项
LDFLAGS   # 链接选项
MAKE      # 构建工具命令
```
为变量赋值：
```Makefile
var  = value  # 允许递归
var := value  # 禁止递归
var += value  # 在 var 的当前值上追加 value
var ?= value  # 若 var 为空，则赋值为 value
```
使用变量的值：
```Makefile
$(CC)
$(CXX)
```
用特殊符号表示的常用值：
```Makefile
$@     # 当前 targets
$<     # 第一个 prerequisite
$?     # 更新时间晚于当前 targets 的 prerequisites
$^     # 所有的 prerequisites, 用空格分隔
$(@D)  # 当前 targets 所在的 directory
$(@F)  # 当前 targets 所在的 file
$(<D)  # 第一个 prerequisite 所在的 directory
$(<F)  # 第一个 prerequisite 所在的 file
```

### 通配符
`%` 表示 *for each*，例如：

```Makefile
OBJS = main.o library.o
$(OBJS) : %.o : %.c
    $(CC) -c $(CFLAGS) $< -o $@
```
相当于
```Makefile
main.o : main.c
    $(CC) -c $(CFLAGS) main.c -o main.o
library.o : library.c
    $(CC) -c $(CFLAGS) library.c -o library.o
```

### 示例
以《[手动构建](#手动构建)》中的项目为例，其构建过程可以写进 [`Makefile`](./make/Makefile)。

# CMake<a href id="CMake"></a>

## 参考资料
### 官方文档
- [帮助文档](https://cmake.org/cmake/help/latest/)
  - [cmake(1)](https://cmake.org/cmake/help/latest/manual/cmake.1.html)：命令行界面程序
  - [ccmake(1)](https://cmake.org/cmake/help/latest/manual/ccmake.1.html)：文字形式的“图形界面”程序
  - [cmake-buildsystem(7)](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html)：系统配置

### 入门教程
- 《[CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial)》provides a step-by-step tutorial covering common build system use cases that CMake helps to address.
- 《[Programming in C++](https://www.ece.uvic.ca/~frodo/cppbook/)》by Michael Adams from University of Victoria
  - Lecture Slides [(Version 2019-02-04)](https://www.ece.uvic.ca/~frodo/cppbook/downloads/lecture_slides_for_programming_in_c++-2019-02-04.pdf)
  - Video Presentations (YouTube)
    - Build Tools (Make and CMake)
      - [Introduction](https://youtu.be/FPcK_swg-f8)
      - [Make --- Introduction](https://youtu.be/FsGAM2pXP_Y)
      - [CMake --- Introduction](https://youtu.be/Ak6cGZshduY)
      - [CMake --- Examples](https://youtu.be/cDWOECgupDg)

## 术语
- **源文件目录 (source dir)**、**源文件树 (source tree)**：项目根目录，必须含有一个 `CMakeLists.txt` 文件。
- **构建目录 (build dir)**、**构建树 (build tree)**、**二进制树 (binary tree)**：存放构建产物（目标文件、库文件、可执行文件）的目录。
- **内部构建 (in-source build)**：在源文件目录下构建（⚠️ 会污染源文件目录）。
- **外部构建 (out-of-source build)**：在源文件目录外构建 👍。
- **构建配置 (build configuration)**：由一组构建工具（编译器、链接器）的配置选项所构成的构建参数集。

## `cmake` 命令<a href id="cmake-cmd"></a>
CMake 参与的构建过程可以分为以下几个阶段：
1. CMake 读取 `CMakeLists.txt` 文件，生成**本地构建工具 (native build tool)** (e.g. [`make`](#make-cmd)) 所需的**本地构建文件 (native build file)** (e.g. [`Makefile`](#Makefile))：
   ```shell
   cmake [<options>] -S <source-dir> -B <build-dir> # cmake 3.13.5+ 推荐用法
   cmake [<options>] <source-dir>         # 建议用 -S <source-dir>
   cmake [<options>] <existing-build-dir> # 建议用 -B <existing-build-dir>
   ```
2. *本地构建工具*读取*本地构建文件*，调用**本地工具链 (native tool chain)** 进行构建。这一步可借助 CMake 以跨平台的方式来完成：
   ```shell
   cmake --build <build-dir> [<options>] [-- <build-tool-options>]
   ```
3. 安装到默认或指定路径：
   ```shell
   # 安装当前目录下的目标到默认路径
   cmake --install .
   # 安装当前目录下的目标到指定路径
   cmake --install . --prefix <installdir>
   ```

### 选项
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
# 指定 source-dir 和 build-dir，需要 cmake 3.13.5+
cmake -S <source-dir> -B <build-dir>
```

### 示例
```shell
cd ${SOURCE_DIR} # ./
BUILD_TYPE=Debug  # 或 Release、RelWithDebInfo、MinSizeRel
mkdir -p _build/$BUILD_TYPE
cd _build/$BUILD_TYPE
cmake -S ../.. -B . -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
      -D CMAKE_C_COMPILER=$(which gcc) -D CMAKE_CXX_COMPILER=$(which g++)
```

## `CMakeLists.txt` 文件<a href id="CMakeLists"></a>
`CMakeLists.txt` 是驱动 CMake 程序运行的*脚本文件*，它由**命令 (command)** 和**注释 (comment)** 组成：

- 命令的名称*不区分大小写*，形式上与函数调用类似。
- 命令的操作对象称为 **CMake 变量 (CMake variable)**，其名称*区分大小写*。
- 注释一般以 `#` 开始，至行尾结束。

完整的语法定义参见《[cmake-language(7)](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)》。

### 命令
完整的 CMake 命令列表参见《[cmake-commands(7)](https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html)》。

设置项目所允许的最低版本：
```cmake
cmake_minimum_required(VERSION 3.0)
```

设置项目信息：
```cmake
project(<PROJECT-NAME> [<language-name>...])
project(<PROJECT-NAME>
        [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
        [DESCRIPTION <project-description-string>]
        [HOMEPAGE_URL <url-string>]
        [LANGUAGES <language-name>...])
```

创建供用户设置的可选项：
```cmake
option(<variable> "<help_text>" [value])
```

添加头文件搜索路径：
```cmake
include_directories([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])
```

添加含有子项目的子目录：
```cmake
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```

向终端输出信息：
```cmake
message([<mode>] "message to display" ...)
# 例如 (自动添加换行符);
message("hello, world")
```

设置 CMake 变量的值：
```cmake
# 设置局部变量:
set(<variable> <value>... [PARENT_SCOPE])
# 设置缓存变量:
set(<variable> <value>... CACHE <type> <docstring> [FORCE])
# 设置环境变量:
set(ENV{<variable>} [<value>])
```

### 变量
完整的 CMake 变量列表参见《[`cmake-variables(7)`](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html)》。

### 查找

```cmake
# 将头文件 name.h 所在文件夹的完整路径存入 VAR 中：
find_path (<VAR> name.h [path1 path2 ...])
# 将库文件 libname.[a|so|dylib] 的完整路径存入 VAR 中：
find_library (<VAR> name [path1 path2 ...])
```

### 目标

添加构建*可执行文件*的目标：
```cmake
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
```

添加构建*库*的目标：
```cmake
add_library(<name> [STATIC | SHARED | MODULE]
            [EXCLUDE_FROM_ALL]
            [source1] [source2 ...])
```

以上命令的第一个参数 `<name>` 表示被创建目标的*逻辑名*，必须（在整个 CMake 项目内）全局唯一；实际被构建的文件名为*物理名*或*输出名*，不必全局唯一。默认情况下，输出名*等于*逻辑名，但可以通过设置 `OUTPUT_NAME` 来改变：

```cmake
add_executable(test_algebra_matrix matrix.cpp)  # 逻辑名为 test_algebra_matrix
set_target_properties(test_algebra_matrix PROPERTIES OUTPUT_NAME matrix)  # 输出名为 matrix
```

### 链接
一般形式：
```cmake
# 为当前 CMakeLists.txt 剩余部分及子目录中的所有目标设置链接项目：
link_libraries([item1 [item2 [...]]] [[debug|optimized|general] <item>] ...)
# 为特定目标设置链接项目：
target_link_libraries(<target> ... <item>... ...)
```
其中
- `target` 必须是以 `add_executable` 或 `add_library` 命令添加的目标。
- `item` 可以是
  - 当前项目的库目标
  - 某个库文件的完整路径
  - 某个库文件的文件名
  - 链接选项

### 示例
依然以《[手动构建](#手动构建)》中的项目为例。
创建三个 `CMakeLists.txt` 文件：

```shell
.
├── CMakeLists.txt  # 用于构建整个项目
├── include
│   └── math.h
├── src
│   ├── CMakeLists.txt  # 用于构建 `lib_math`
│   └── math.c
└── test
    ├── CMakeLists.txt  # 用于构建 `test_math`
    └── math.c
```

## CMake Tools<a href id="CMake-Tools"></a>

微软发布的代码编辑器 [Visual Studio Code](https://code.visualstudio.com/) 具有*体量轻*、*易扩展*、*多语言*、*跨平台*等优点，利用各种[扩展](https://marketplace.visualstudio.com/)，很容易将其改造为多语言共用的**集成开发环境 (Integrated Development Environment, IDE)**。本节介绍利微软提供的 [CMake Tools](https://vector-of-bool.github.io/docs/vscode-cmake-tools/) 扩展。

### 工具集

工具集（编译器组合）既可由自动扫描获得，也可在 `~/.local/share/CMakeTools/cmake-tools-kits.json` 中手动添加：

```json
[
  {
    "name": "Clang 12.0.0",
    "compilers": {
      "C": "/usr/bin/clang",
      "CXX": "/usr/bin/clang++"
    }
  },
  {
    "name": "GCC 11.2.0 x86_64-apple-darwin20",
    "compilers": {
      "C": "/usr/local/bin/x86_64-apple-darwin20-gcc-11",
      "CXX": "/usr/local/bin/x86_64-apple-darwin20-g++-11"
    }
  }
]
```

### 构建

1. 用 VS Code 打开一个 CMake 项目。
1. 平行于顶层 `CMakeLists.txt` 创建名为 `.vscode` 的目录（`vscode` 前面的 `.` 不能遗漏），并在其中创建名为 `settings.json` 的文件，用于设定构建目录等全局配置项。
   - 本节示例 [`./.vscode/settings.json`](./make/.vscode/settings.json) 设置了 `cmake.buildDirectory`（构建目录）及  `cmake.generator`（构建工具）两个变量。
   - 完整变量及选项列表参见《[Configuring CMake Tools](https://vector-of-bool.github.io/docs/vscode-cmake-tools/settings.html)》。
1. 在 VS Code 底部的**状态栏 (status bar)** 中：
   - 单击 `CMake` 按钮，从顶部弹出的四种**构建类型 (build type)** 中任选一种，单击之以完成**配置 (configure)**。若对某些 CMake 选项的默认值不满意，可在生成的 `${BUILD_DIR}/CMakeCache.txt` 文件中直接修改。
   - 单击 `Build` 按钮，完成**构建 (build)**。默认构建目标为 `Build` 按钮右侧的 `all`，可单击之以选择其他目标。


### 调试

1. 在 `.vscode` 中创建名为 `launch.json` 的文件，用于控制可执行文件的运行及调试。
   - 本节示例 [`./.vscode/launch.json`](./make/.vscode/launch.json) 基本照搬了《[Target Debugging and Launching](https://vector-of-bool.github.io/docs/vscode-cmake-tools/debugging.html)》中的示例，只修改了个别选项的值。
1. 在 VS Code 中打开源文件。单击行号左侧以设置**断点 (breakpoint)**。按功能键 `F5` 启动调试。
   - ⚠️ [用状态栏中的 🐞 键启动调试可能出错。](https://github.com/microsoft/vscode-cmake-tools/issues/506#issuecomment-410021984)
1. 在命令行环境中，亦可用《[断点调试](./debug.md)》中介绍的 GDB / LLDB 命令进行调试。
   - 此方法不依赖于本节介绍的 VS Code 及 CMake Tools。

### 测试

若项目配置了 [CTest](./unittest.md#CTest)，则启用 CMake Tools 后，可一键运行所有测试。

若遇到环境变量与交互式 shell 不一致的问题，可通过在 VS Code 的 Settings 中搜索 `ctest`，在 `Test Environment` 中设置相应环境变量解决，例如：

```shell
PATH=/home/user/.local/bin:/usr/local/bin:/usr/bin
TERM=xterm-256color
```

# Ninja<a href id="Ninja"/>

## 参考资料

- [Manual](https://ninja-build.org/manual.html)

## 安装

- macOS

  ```shell
  brew install ninja
  ninja --version
  ```

- Ubuntu

  ```shell
  sudo apt install ninja-build
  ninja --version
  ```

## `ninja` 命令<a href id="ninja-cmd"></a>

### 选项

```
usage: ninja [options] [targets...]

if targets are unspecified, builds the 'default' target (see manual).

options:
  --version      print ninja version ("1.10.0")
  -v, --verbose  show all command lines while building

  -C DIR   change to DIR before doing anything else
  -f FILE  specify input build file [default=build.ninja]

  -j N     run N jobs in parallel (0 means infinity) [default=3 on this system]
  -k N     keep going until N jobs fail (0 means infinity) [default=1]
  -l N     do not start new jobs if the load average is greater than N
  -n       dry run (don't run commands but act like they succeeded)

  -d MODE  enable debugging (use '-d list' to list modes)
  -t TOOL  run a subtool (use '-t list' to list subtools)
    terminates toplevel options; further flags are passed to the tool
  -w FLAG  adjust warnings (use '-w list' to list warnings)
```

### 示例

```shell
cd ${SOURCE_DIR} # ./
BUILD_TYPE=Debug  # 或 Release、RelWithDebInfo、MinSizeRel
mkdir -p _build/${BUILD_TYPE}
cd _build/${BUILD_TYPE}
cmake -G Ninja -S ../.. -B . -D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -D CMAKE_C_COMPILER=$(which gcc-9) -D CMAKE_CXX_COMPILER=$(which g++-9)
ninja
ninja install
ninja clean
```

## `build.ninja` 文件<a href id="build.ninja"></a>

用于驱动 [`ninja` 命令](#ninja-cmd)运行的脚本文件，类似于 [`Makefile` 文件](#Makefile)。

⚠️ 不要手写！用 [CMake](#CMake) 生成！

