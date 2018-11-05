# Gmsh 简介

[Gmsh](http://gmsh.info/) 是一款带有简单 CAD 和后处理功能的三维有限元网格生成软件.
在遵循 [GPL](http://gmsh.info/#Licensing) 条款的前提下, 用户可以修改或重新发布其[源代码](https://gitlab.onelab.info/gmsh/gmsh).
初学者可以直接下载各种操作系统下的[预编译版](http://gmsh.info/bin/).

## 使用方式
Gmsh 可以按三种方式来使用: GUI 互动程序, 脚本驱动程序, C++ 程序库.
其中, 脚本驱动模式是学习价值最高的一种:
- 绝大多数 GUI 功能都有对应的脚本命令, 每一条 GUI 操作都会被记录在脚本文件中;
- 在 GUI 中可以很容易地打开或加载脚本文件:
```
Modules
    Geometry
        Reload script
        Edit script
```
> 建议: 在脚本文件中定义简单几何实体, 在 GUI 中执行选择或变换等更加复杂的操作.

在终端中, 可以直接令 Gmsh 完成网格生成和输出操作:
```shell
gmsh t1.geo -2
```
常用命令行参数:

| 参数 | 功能 |
| ---- | ---- |
| `-1`/`-2`/`-3` | 生成一/二/三维网格 |
| `-o filename` | 将网格输出到指定文件 |
| `-format string` | 选择网格文件格式, 例如 `msh4`, `msh2`, `vtk` |
| `-bin` | 以二进制模式输出 |
| `-part n` | 将网格分割为 `n` 块 (用于并行计算) |

完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [3.3 Command-line options](http://gmsh.info/doc/texinfo/gmsh.html#Command_002dline-options).

## GEO 文件说明
Gmsh 自定义了一种脚本语言, 用各种命令驱动主程序执行
- 构造几何对象
- 划分网格
- 按指定格式输出
等功能. 这些命令以字符形式存储于 GEO 文件 (即 `.geo` 文件) 中. 当 GEO 文件被加载时, *文本解析器 (parser)* 会将字符形式的命令解析到对应的可执行代码.
> GEO 命令的语法与 C++ 较为接近. 在代码编辑器(如 [Visual Studio Code](https://code.visualstudio.com)) 中, 将 GEO 文件的语言设置为 C++, 可以高亮显示一些信息, 有助于提高可读性.

[Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 采用如下符号约定:

1. 关键词用 `UpperCamelCase` 或 `Upper Camel Case` 表示.
2. 变量用 `lowerCamelCase` 表示.
3. 变量与定义之间用 `:` 分隔.
4. 可选项置于 `< >` 中.
5. 可替换项用 `|` 分隔.
6. 重复项用 `...` 表示.

### 通用命令及选项
在这里, *通用*指的是某项功能不专属于几何/网格/求解器/后处理模块.

#### 注释
注释是给人阅读的辅助信息, parser 会忽略这些内容.
GEO 文件里的注释采用 C++ 风格:

- `//` 后一行以内的内容均为注释;
- 从 `/*` 到 `*/` 之间的所有内容均为注释.

除注释以外, 所有空白字符 (空格 `' '`, 制表符 `\t`, 换行符 `\n`) 也都被 parser 忽略.

#### 表达式
GEO 表达式的取值有两种类型: 字符型, 浮点型 (没有整型).
对于计算结果有确定取值类型的表达式, 可以根据计算结果的类型将其归为*浮点型表达式 (Floating Point Expressions)* 或*字符型表达式 (Character Expressions)*.
此外, 还有一种计算结果类型不定的表达式, 专门用于表示颜色信息, 因此称为*颜色表达式 (Color Expressions)*.

##### 浮点型表达式
符号后接 `~{floatExpr}` 表示用 `_` 将该符号和 `floatExpr` 的计算结果*串接 (concatenate)* 起来. 例如:
```cpp
For i In {1:3}
    x~{i} = i;
EndFor
```
其中 `i` 的取值为 `1`, `2`, `3`, 所以 `x~{i}` 等价于 `x_1`, `x_2`, `x_3`, 而上述循环等价于:
```cpp
x_1 = 1;
x_2 = 2;
x_3 = 3;
```

`[]` 用于从列表中抽取一项, `#` 用于获取列表长度.

`Entity{:}` 表示提取*所有*同类实体. 其中 `Entity` 可以是 `Point`, `Curve`, `Surface`, `Volume` 之一.

预定义的浮点型表达式:
```cpp
Pi  // 3.1415926535897932.
GMSH_MAJOR_VERSION  // Gmsh 主版本号
GMSH_MINOR_VERSION  // Gmsh 次版本号
GMSH_PATCH_VERSION  // Gmsh 补丁版本号
MPI_Size  // 总进程数, 通常为 1, 除非编译时设置了 ENABLE_MPI
MPI_Rank  // 当前进程的 rank
Cpu          // 当前 CPU 时间, 单位为 second
Memory       // 当前内存用量, 单位为 MB
TotalMemory  // 可用内存总量, 单位为 MB
newp    // 下一个可用的 Point tag
newl    // 下一个可用的 Curve tag, 较早的版本里将 Curve 称为 Line, 因此这里为 l
news    // 下一个可用的 Surface tag
newv    // 下一个可用的 Volume tag
newll   // 下一个可用的 Curve Loop tag, 较早的版本里将 Curve 称为 Line, 因此这里为 l
newsl   // 下一个可用的 Surface Loop tag
newreg  // 下一个可用的 REGion tag, 即 max(new*, physicalTags)
```

##### 字符型表达式
预定义的字符型表达式:
```cpp
Today  // 以字符形式表示的当前日期
GmshExecutableName  // 当前所用 Gmsh 可执行文件的完整路径
CurrentDirectory | CurrentDir  // 当前 GEO 文件所在目录
```
预定义的返回字符型结果的函数:
```cpp
// 提取文件名前缀, 即除去扩展名:
StrPrefix("hello.geo")  // 返回 "hello"
// 提取相对路径, 即除去文件名前的路径:
StrRelative("/usr/bin/gcc")  // 返回 "gcc"
// 串接字符串:
StrCat("hello", ", ", "world")  // 返回 "hello, world"
// 串接字符串, 但字符串之间添加换行符 '\n'
Str("hello", ", ", "world")     // 返回 "hello\n, \nworld"
// 根据第一个表达式的值是否为零, 输出后两个字符串之一:
StrChoice(1, "hello", "world")  // 返回 "hello"
StrChoice(0, "hello", "world")  // 返回 "world"
// 提取子字符串:
StrSub("hello, world", 7, 11)  // 返回 "world"
StrSub("hello, world", 7)      // 返回 "world"
// 转换为大写形式:
UpperCase("hello, world")  // 返回 "HELLO, WORLD"
// 类似于 C 标准库函数 sprintf:
Sprintf("%g", Pi)  // 返回 "3.14159"
// 获取环境变量的值:
GetEnv("HOME")  // 返回当前用户家目录绝对路径
// 子字符串替换:
StrReplace("hello, world", "o", "O")  // 返回 "hellO, wOrld"
// 其他一些不常用的函数:
AbsolutePath(filename) 
DirName(filename)
GetString(charExpr<, charExpr>)
GetStringValue(charExpr, charExpr)
NameToString(string)
N2S(string)
DefineString(charExpr, onelabOptions)
```

##### 颜色表达式
*颜色表达式 (Colors Expressions)* 用于表示颜色信息, 可以是以下任意一种形式:
```cpp
colorName
{red, green, blue}  // [0, 255] 之间的整数, 表示红绿蓝分量数值
{red, green, blue, alpha}  // 前三个同上, 最后一个表示透明度
colorOption
```

#### 运算符
GEO 文件里的运算符与 C/C++ 里的同名运算符类似.
但有一个例外: 这里的*逻辑或 (logical or)* 运算符 `||` 总是会对其两侧的表达式求值; 而在 C/C++ 里, 只要第一个表达式的值为 `true`, 则不会对第二个表达式求值.

运算符优先级:
1. `()`, `[]`, `.`, `#`
2. `^`
3. `!`, `++`, `--`, `-` (单目)
4. `*`, `/`, `%`
5. `+`, `-` (双目)
6. `<`, `>`, `<=`, `>=`
7. `==`, `!=`
8. `&&`
9. `||`
10. `?:`
11. `=`, `+=`, `-=`, `*=`, `/=`

#### 内置函数
所有函数名的首字母均为大写, 除以下几个函数以 `F` 为首字母外, 其余函数均为其本名 (如 `Sin`, `Cos`, `Tan`):

| 函数 | 功能 |
| ---- | ---- |
| `Fabs(x)` | 绝对值 |
| `Fmod(x, y)` | `x % y`, 结果与 `x` 同号 |

完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [Built-in functions](http://gmsh.info/doc/texinfo/gmsh.html#Built_002din-functions).

#### 自定义宏
暂时不用.

#### 控制流
从 `begin` 到 `end`, 含起止项, 步进为 `step` (默认值为 `1`):
```cpp
// 不使用循环指标时:
For (begin : end : step)
    ...
EndFor
// 使用循环指标时:
For i In {begin : end : step}
    ...
EndFor
```

条件分支:
```cpp
If (condition)
    ...
ElseIf (condition)
    ...
Else
    ...
EndIf
```

#### 通用命令
一些常用命令:
```cpp
Abort;  // 中断解析当前脚本
Exit;   // 退出 Gmsh
Merge filename;  // 将指定文件中的数据合并到当前模型
// 相当于 C 标准库函数 printf:
Printf("%g", Pi);  // 在终端或 GUI 信息栏输出 "3.14159"
Printf("%g", Pi) > "temp.txt";  // 输出到指定文件
```
完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [4.7 General commands](http://gmsh.info/doc/texinfo/gmsh.html#General-commands).

#### 通用选项
暂时不用.


## Running

## General Tools

## Geometry Module

## Mesh Module

## Post-processing Module

## File Formats

