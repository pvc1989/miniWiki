# 构建工具

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
