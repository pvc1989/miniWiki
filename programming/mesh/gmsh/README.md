---
title: Gmsh
---

[Gmsh](http://gmsh.info/) 是一款带有简单 CAD 和后处理功能的三维有限元网格生成软件。
在遵守 [GPL](http://gmsh.info/#Licensing) 条款的前提下，用户可以修改或重新发布其[源代码](https://gitlab.onelab.info/gmsh/gmsh)。
初学者可以直接下载运行[预编译版](http://gmsh.info/bin/)。

本文档是基于《[Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html)》编写的阅读笔记，不是原文的完整翻译，也没有严格遵循原文的结构。部分术语保留英文是为了方便读者查阅原始文档。

Gmsh 可以按三种方式来使用：
1. 人机互动的 GUI 程序
2. 脚本驱动的 CLI 程序
3. [C++、C、Python、Julia 程序库](http://gmsh.info/doc/texinfo/gmsh.html#Gmsh-application-programming-interface)

其中*脚本驱动程序*是学习性价比最高的一种：
- 绝大多数 GUI 功能都有对应的脚本命令，每一条 GUI 操作都会被记录在脚本文件中。
- 在 GUI 中可以很容易地打开或加载脚本文件：
  ```
  Modules
    Geometry
      Reload script
      Edit script
  ```
- 建议：在脚本文件中定义简单几何实体，在 GUI 中执行*选择*或*变换*等更加复杂的操作。

# 命令行参数

在终端中，可以直接令 Gmsh 完成网格生成和输出操作：

```shell
gmsh t1.geo -2
```
常用命令行参数：

| 参数 | 功能 |
| ---- | ---- |
| `-1`/`-2`/`-3` | 生成相应维数的网格 |
| `-o filename` | 将网格输出到指定文件 |
| `-format string` | 选择网格文件格式，例如 `msh4`、`msh2`、`vtk` |
| `-bin` | 以二进制模式输出 |
| `-part n` | 将网格分割为 `n` 块 (用于并行计算) |

详见《[Gmsh command-line interface](http://gmsh.info/doc/texinfo/gmsh.html#Gmsh-command_002dline-interface)》。

# 脚本语法

Gmsh 自定义了一种脚本语言，用各种命令驱动主程序完成
- 几何建模
- 网格生成
- 网格输出

这些命令以字符形式存储于 GEO 文件（即 `.geo` 文件）中。当 GEO 文件被加载时，**文本解析器 (parser)** 会将字符形式的命令解析到对应的可执行代码。
GEO 命令的语法与 C++ 较为接近。在代码编辑器（如 [Visual Studio Code](https://code.visualstudio.com)）中，将 GEO 文件的语言设置为 C++，可以高亮显示一些信息，有助于提高可读性。

《[Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html)》采用如下符号约定：

1. 关键词用 `UpperCamelCase` 或 `Upper Camel Case` 表示。
2. 变量用 `lowerCamelCase` 表示。
3. 变量与定义之间用 `:` 分隔。
4. 可选项置于 `< >` 中。
5. 可替换项用 `|` 分隔。
6. 重复项用 `...` 表示。

## 通用命令及选项
在这里，*通用*指的是某项功能不专属于几何、网格、求解器、后处理模块。

### 注释
注释是给人阅读的辅助信息，parser 会忽略这些内容。
GEO 文件里的注释采用 C++ 风格：

- `//` 后一行以内的内容均为注释；
- 从 `/*` 到 `*/` 之间的所有内容均为注释。

除注释以外，所有空白字符（空格 `' '`，制表符 `\t`，换行符 `\n`）也都被 parser 忽略。

### 表达式
GEO 表达式的取值有两种类型：字符型，浮点型（没有整型）。
对于计算结果有确定取值类型的表达式，可以根据计算结果的类型将其归为**浮点型表达式 (Floating Point Expressions)** 或**字符型表达式 (Character Expressions)**。
此外，还有一种计算结果类型不定的表达式，专门用于表示颜色信息，因此称为**颜色表达式 (Color Expressions)**。

#### 浮点型表达式
符号后接 `~{floatExpr}` 表示用 `_` 将该符号和 `floatExpr` 的计算结果**串接 (concatenate)** 起来。例如：
```cpp
For i In {1:3}
    x~{i} = i;
EndFor
```
其中 `i` 的取值为 `1`、`2`、`3`，所以 `x~{i}` 等价于 `x_1`、`x_2`、`x_3`，而上述循环等价于：
```cpp
x_1 = 1;
x_2 = 2;
x_3 = 3;
```

`[]` 用于从列表中抽取一项，`#` 用于获取列表长度。

`Entity{:}` 表示提取*所有*同类实体。其中 `Entity` 可以是 `Point`、`Curve`、`Surface`、`Volume` 之一。

预定义的浮点型表达式：
```cpp
Pi  // 3.1415926535897932
GMSH_MAJOR_VERSION  // Gmsh 主版本号
GMSH_MINOR_VERSION  // Gmsh 次版本号
GMSH_PATCH_VERSION  // Gmsh 补丁版本号
MPI_Size  // 总进程数，通常为 1, 除非编译时设置了 ENABLE_MPI
MPI_Rank  // 当前进程的 rank
Cpu          // 当前 CPU 时间，单位为 second
Memory       // 当前内存用量，单位为 MB
TotalMemory  // 可用内存总量，单位为 MB
newp    // 下一个可用的 Point tag
newl    // 下一个可用的 Curve tag，较早的版本里将 Curve 称为 Line，因此这里为 l
news    // 下一个可用的 Surface tag
newv    // 下一个可用的 Volume tag
newll   // 下一个可用的 Curve Loop tag，较早的版本里将 Curve 称为 Line，因此这里为 l
newsl   // 下一个可用的 Surface Loop tag
newreg  // 下一个可用的 REGion tag，即 max(new*，physicalTags)
```

#### 字符型表达式
预定义的字符型表达式：
```cpp
Today  // 以字符形式表示的当前日期
GmshExecutableName  // 当前所用 Gmsh 可执行文件的完整路径
CurrentDirectory | CurrentDir  // 当前 GEO 文件所在目录
```
预定义的返回字符型结果的函数：
```cpp
// 提取文件名前缀，即除去扩展名：
StrPrefix("hello.geo")  // 返回 "hello"
// 提取相对路径，即除去文件名前的路径：
StrRelative("/usr/bin/gcc")  // 返回 "gcc"
// 串接字符串：
StrCat("hello", ", ", "world")  // 返回 "hello, world"
// 串接字符串，但字符串之间添加换行符 '\n'
Str("hello", ", ", "world")     // 返回 "hello\n, \nworld"
// 根据第一个表达式的值是否为零，输出后两个字符串之一：
StrChoice(1, "hello", "world")  // 返回 "hello"
StrChoice(0, "hello", "world")  // 返回 "world"
// 提取子字符串：
StrSub("hello, world", 7, 11)  // 返回 "world"
StrSub("hello, world", 7)      // 返回 "world"
// 转换为大写形式：
UpperCase("hello, world")  // 返回 "HELLO, WORLD"
// 类似于 C 标准库函数 sprintf：
Sprintf("%g", Pi)  // 返回 "3.14159"
// 获取环境变量的值：
GetEnv("HOME")  // 返回当前用户家目录绝对路径
// 子字符串替换：
StrReplace("hello, world", "o", "O")  // 返回 "hellO, wOrld"
// 其他一些不常用的函数：
AbsolutePath(filename) 
DirName(filename)
GetString(charExpr<, charExpr>)
GetStringValue(charExpr, charExpr)
NameToString(string)
N2S(string)
DefineString(charExpr, onelabOptions)
```

#### 颜色表达式
**颜色表达式 (Colors Expressions)** 用于表示颜色信息，可以是以下任意一种形式：
```cpp
colorName
{red, green, blue}  // [0, 255] 之间的整数，表示红绿蓝分量数值
{red, green, blue, alpha}  // 前三个同上，最后一个表示透明度
colorOption
```

### 运算符
GEO 文件里的运算符与 C/C++ 里的同名运算符类似。
但有一个例外：这里的**逻辑或 (logical or)** 运算符 `||` 总是会对其两侧的表达式求值；而在 C/C++ 里，只要第一个表达式的值为 `true`，则不会对第二个表达式求值。

运算符优先级：
1. `()`, `[]`, `.`, `#`
2. `^`
3. `!`, `++`, `--`, `-`（单目）
4. `*`, `/`, `%`
5. `+`, `-`（双目）
6. `<`, `>`, `<=`, `>=`
7. `==`, `!=`
8. `&&`
9. `||`
10. `?:`
11. `=`, `+=`, `-=`, `*=`, `/=`

### 内置函数
所有函数名的首字母均为大写，除以下几个函数以 `F` 为首字母外，其余函数均为其本名（如 `Sin`、`Cos`、`Tan`）：

| 函数 | 功能 |
| ---- | ---- |
| `Fabs(x)` | 绝对值 |
| `Fmod(X, y)` | `x % y`，结果与 `x` 同号 |

详见《[Built-in functions](http://gmsh.info/doc/texinfo/gmsh.html#Built_002din-functions)》。

### 自定义宏
暂时不用。

### 控制流
从 `begin` 到 `end`，含起止项，步进为 `step`（默认值为 `1`）：
```cpp
// 不使用循环指标时：
For (begin : end : step)
    ...
EndFor
// 使用循环指标时：
For i In {begin : end : step}
    ...
EndFor
```

条件分支：
```cpp
If (condition)
    ...
ElseIf (condition)
    ...
Else
    ...
EndIf
```

### 通用命令
一些常用命令：
```cpp
Abort;  // 中断解析当前脚本
Exit;   // 退出 Gmsh
Merge filename;  // 将指定文件中的数据合并到当前模型
// 相当于 C 标准库函数 printf：
Printf("%g", Pi);  // 在终端或 GUI 信息栏输出 "3.14159"
Printf("%g", Pi) > "temp.txt";  // 输出到指定文件
```

详见《[General scripting commands](http://gmsh.info/doc/texinfo/gmsh.html#General-scripting-commands)》

### 通用选项
暂时不用。

## 几何模块
### CAD 内核
CAD 内核可以在 GEO 文件头部通过以下命令之一设定：
```cpp
SetFactory("Built-in");
SetFactory("OpenCASCADE");
```

第一种为 Gmsh 自带的简易 CAD 内核，只支持一些简单几何对象的创建和操作。
由于采用了**边界表示法 (Boundary Representation)**，所有几何实体必须**自底向上 (bottom-up)** 创建：
1. 先创建 `Point`，
2. 再以 `Point`s 为边界或控制点创建 `Curve`，
3. 再以 `Curve`s 为边界创建 `Surface`，
4. 最后以 `Surface`s 为边界创建 `Volume`。

第二种为开源 CAD 系统 [OpenCASCADE](https://www.opencascade.com) 的内核，支持一些高级几何对象的创建和操作。
如果没有特殊的需求，建议使用这种内核。

### 创建初等实体
只具有几何意义的对象称为**初等实体 (elementary entity)**。
初等实体在创建时，被赋予一个正整数（*非正整数*为系统保留）编号，《[Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html)》称其为**标签 (tag)**。
这些 `tag` 满足：
- 每个初等 `Point` 具有唯一的 `tag` 
- 每个初等 `Curve` 具有唯一的 `tag` 
- 每个初等 `Surface` 具有唯一的 `tag` 
- 每个初等 `Volume` 具有唯一的 `tag` 

多数命令的语法与 C++ 相同，尤其要注意：每个语句最后的 `;` 不能省略。

- 圆括号 `()` 中的编号表示*创建*一个新的实体。
- 花括号 `{}` 中的编号表示*引用*一个已有的实体。
- 尖括号 `<>` 中的内容为可选项。

#### `Point`s
创建三维点：
```cpp
Point(pointTag) ={
    x, y, z  // 直角坐标分量
    <, /* 单元尺寸 */elementSize>
};
```

#### `Curve`s
通过连接两点创建直线段：
```cpp
Line(curveTag) = {startPointTag, endPointTag};
```

通过一组点创建样条曲线：
```cpp
 Bezier(curveTag) = {pointTagList};
BSpline(curveTag) = {pointTagList};
 Spline(curveTag) = {pointTagList};
```

创建圆弧：
```cpp
// Built-in 或 OpenCASCADE 均可。
// 如果使用 Built-in 内核，则弧度必须严格小于 Pi：
Circle(curveTag) ={
    startPointTag,
    centerPointTag,
    endPointTag
};
// 必须用 OpenCASCADE 内核：
Circle(curveTag) ={
    centerX, centerY, centerZ,
    radius<, startAngle, endAngle>
};
```

#### `Surface`s
如果使用 built-in 内核，必须按以下流程：
```cpp
// 先创建一个或多个“曲线环 (Curve Loop)”：
Curve Loop(curveLoopTag) = {curveTagList};
// 再创建平面区域：
Plane Surface(surfaceTag) = {curveLoopTagList};
```
其中，第一个曲线环表示外边界，其余曲线环表示内边界。一个有效的曲线环必须满足：
- 封闭;
- 列表中的曲线**有序 (ordered)** 并且具有相同**取向 (oriented)**，曲线标签前加负号表示反向。

如果使用 OpenCASCADE 内核，可以通过以下命令快速创建平面区域：
```cpp
Disk(surfaceTag) = {
    centerX, centerY, centerZ,
    radius  // 圆
};
Disk(surfaceTag) = {
    centerX, centerY, centerZ,
    radiusX, radiusY  // 椭圆
};
Rectangle(surfaceTag) = {
    cornerX, cornerY, cornerZ,  // 左下角
    width, height<, /* 圆角半径 */radius>
};
```

#### `Volume`s
如果使用 built-in 内核，必须按以下流程：
```cpp
// 先创建一个或多个“曲面环 (Surface Loop)”：
Surface Loop(surfaceLoopTag) = {surfaceTagList};
// 再创建空间区域（三维流形）：
Volume(volumeTag) = {surfaceLoopTagList};
```
其中，第一个曲面环表示外边界，其余曲面环表示内边界。一个有效的曲面环必须满足：
- 封闭；
- 列表中的曲面**有序 (ordered)** 并且具有相同**取向 (oriented)**，曲面标签前加负号表示反向。

如果使用 OpenCASCADE 内核，可以通过以下命令快速创建空间区域（三维流形）：
```cpp
Sphere(volumeTag) = { 
    centerX, centerY, centerZ,
    radius
};
Box(volumeTag) = {
    cornerX, cornerY, cornerZ,
    dX, dY, dZ
};
Cylinder(volumeTag) = { 
    centerX, centerY, centerZ,
    axisX, axisY, axisZ,
    radius<, angle>
};
Torus(volumeTag) = { 
    centerX, centerY, centerZ,
    radiusOuter, radiusInner<, angle>
};
Cone(volumeTag) = { 
    centerX, centerY, centerZ,
    axisX, axisY, axisZ,
    radiusOuter, radiusInner<, angle>
};
Wedge(volumeTag) = { 
    cornerX, cornerY, cornerZ,
    dX, dY, /* 延伸方向 */dZ<, /* 顶面宽度 */topX>
};
```

#### 复合实体
复合实体仍然是几何实体，它由多个具有相同维度的初等实体组成。
在生成网格时，这些初等实体的整体将被视作一个几何实体，即允许一个单元跨越多个初等实体的边界。

```cpp
Compound Curve(curveTag) = {curveTagList};
Compound Surface(surfaceTag) = {surfaceTagList};
```

#### 通过拉伸创建
通过拉伸低维实体来创建高维实体：
```cpp
// 通过 平移 拉伸：
Extrude{
    vectorX, vectorY, vectorZ  // 平移向量
}{
    entityList  // 被拉伸对象
}
// 通过 旋转 拉伸：
Extrude{
    {axisX, axisY, axisZ},     // 旋转轴
    {pointX, pointY, pointZ},  // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
}
// 通过 平移 + 旋转 拉伸：
Extrude{
    {vectorX, vectorY, vectorZ},  // 平移向量
    {axisX, axisY, axisZ},        // 旋转轴
    {pointX, pointY, pointZ},     // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
}
```

### 创建物理实体
一组相同维度的初等实体可以组合成一个**物理实体 (physical entity)**，以便赋予它们物理意义。
例如：材料属性、载荷分布、边界条件等。

每个物理实体也有唯一的 `tag`，这里的*唯一*也是针对同一维度的物理实体而言的。
除此之外，每个物理实体还可以有一个字符串表示的名称。

```cpp
Physical Entity(tag | name<, tag>) <+|->= {entityTagList};
```
这里的 `Entity` 可以是 `Point`、`Curve`、`Surface`、`Volume` 中的任意一个。

### 编辑几何实体

#### 布尔运算
**布尔运算 (Boolean Operation)** 就是将几何区域看作点集的集合运算。
只有 OpenCASCADE 内核支持布尔运算。
所有布尔运算都是通过一条作用在两个实体列表上的指令来完成的：

```cpp
BooleanOperation{passiveEntityList}{toolEntityList}
```

- `BooleanOperation` 代表某种布尔运算，可以是 `BooleanIntersection`、`BooleanUnion`、`BooleanDifference` 之一。
- `passiveEntityList` 代表**被动 (passive)** 实体列表，`toolEntityList` 代表**工具 (tool)** 实体列表，它们可以是

  ```cpp
  <Physical> Curve | Surface | Volume{tagList};  // ; 不能省略
  <... | Delete;>  // 运算完成后删去对应的实体
  ```

新版 Gmsh 支持将运算结果存储为新的实体：
```cpp
BooleanOperation(newEntityTag) = {passiveEntityList}{toolEntityList};
```

示例：
```cpp
SetFactory("OpenCASCADE");
Rectangle(1) = {-5, -5, 0, 10, 10};
Disk(2) = {0, 0, 0, 2};
BooleanDifference(news) = {Surface{1}; Delete;}{Surface{2}; Delete;};
Mesh 2;
```
[demos/boolean](https://gitlab.onelab.info/gmsh/gmsh/tree/master/demos/boolean/) 中有更多示例。

#### 几何变换

```cpp
// 按相同比例放缩：
Dilate{
    {centerX, centerY, centerZ}, 
    factor
}{entityList}
// 按不同比例放缩：
Dilate{
    {centerX, centerY, centerZ},
    {factorX, factorY, factorZ}
}{entityList}
// 旋转：
Rotate{
    {axisX, axisY, axisZ},
    {pointX, pointY, pointZ},
    angle
}{entityList}
// 关于平面对称：
Symmetry{
    A，B，C，D  // A*x + B*y + C*z + D = 0
}{entityList}
// 平移：
Translate{
    {vectorX, vectorY, vectorZ},  // 平移向量
}{entityList}
```
其中 `entityList` 可以是
```cpp
<Physical> Point | Curve | Surface | Volume{tagList}; ...
// 或
Duplicata{<Physical> Point | Curve | Surface | Volume{tagList}; ...};
```

#### 提取边界

```cpp
// 提取边界上低一维的实体，返回其标签：
Boundary{entityList}
// 提取边界上低一维的实体，作为一个复合实体返回其标签：
CombinedBoundary{entityList}
// 提取边界上的点，返回其标签：
PointsOf{entityList}
```

#### 删除

```cpp
删除坐标相同的冗余点：
Coherence;
删除列表中的实体：
<Recursive> Delete{Entity{tagList}; ...};
```
这里的 `Entity` 可以是 `Point`、`Curve`、`Surface`、`Volume` 中的任意一个。
如果列表中的某个实体被列表以外的其他实体所依赖，则不执行 `Delete` 命令。
`Recursive` 表示 `Delete` 命令递归地作用到所有次级实体上。

### 几何选项
详见《[Geometry options](http://gmsh.info/doc/texinfo/gmsh.html#Geometry-options)》。

## 网格模块

### 物理实体对网格的影响
- 如果没有创建物理实体，那么所有单元都会被写入网格文件。
- 如果创建了物理实体，那么
    - 默认情况下，只有物理实体上的单元会被写入网格文件，结点和单元会被重新编号。
    - 通过设置 `Mesh.SaveAll` 选项或使用命令行参数 `-save_all` 可以保存所有单元。

### 设定单元尺寸

⚠️ 自 v4.7.0 起，`CharacteristicLength` 与 `Characteristic Length` 分别被重命名为 `MeshSize` 与 `Mesh Size`。

单元尺寸可以通过以下三种方式设定：
- 如果设定了 `Mesh.CharacteristicLengthFromPoints`，那么可以为每个 `Point` 设定一个**特征长度 (Characteristic Length)**。
- 如果设定了 `Mesh.CharacteristicLengthFromCurvature`，那么网格尺寸将与**曲率 (Curvature)** 相适应。
- 通过背景网格或标量场设定。
这三种方式可以配合使用，Gmsh 会选用最小的一个。

常用命令：
```cpp
// 修改点的特征长度：
Characteristic Length{pointTagList} = length;
```

详见《[Mesh element sizes](http://gmsh.info/doc/texinfo/gmsh.html#Mesh-element-sizes)》。

### 生成结构网格
Gmsh 所生成的网格都是**非结构的 (unstructured)**，即各单元的取向和结点邻接关系完全由其结点列表决定，而不要求相邻单元之间有其他形式的关联，因此不能算是真正意义上的**结构 (structured)** 网格。

所有结构网格单元（四边形、六面体、三棱柱）都是通过合并单纯形（三角形、四面体）而得到的：
```cpp
// 将指定曲面上的 三角形 合并为 四边形：
Recombine Surface{surfaceTagList};
// 将指定空间区域里的 四面体 合并为 六面体 或 三棱柱：
Recombine Volume{volumeTagList};
```

#### 通过拉伸生成
与几何模块中的同名函数类似，只是多了一个 `layers` 参数：
```cpp
// 通过 平移 拉伸：
Extrude{
    vectorX, vectorY, vectorZ  // 平移向量
}{
    entityList  // 被拉伸对象
    layers
};
// 通过 旋转 拉伸：
Extrude{
    {axisX, axisY, axisZ},  // 旋转轴
    {pointX, pointY, pointZ},  // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
    layers
};
// 通过 平移 + 旋转 拉伸：
Extrude{
    {vectorX, vectorY, vectorZ},  // 平移向量
    {axisX, axisY, axisZ},        // 旋转轴
    {pointX, pointY, pointZ},     // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
    layers
};
```
`layers` 用于设定拉伸方式，有以下几种形式可以选择：
```cpp
Layers{elementNumber};  // 拉伸方向的单元数
Layers {                    // 两个列表的长度必须一致
    {elementNumberList},    // 各层的单元数
    {normalizedHeightList}  // 各层的相对高度
};
Recombine Surface{surfaceTagList};  // 将指定曲面上的 三角形 合并为 四边形
Recombine Volume{volumeTagList};    // 将指定空间区域里的 四面体 合并为 三棱柱 或 六面体
```

获取拉伸所得实体的标签：
```cpp
num[] = Extrude {0,0,1} { Surface{1}; Layers{10}; };
// num[0] 为拉伸后的顶面
// num[1] 为拉伸出的空间区域
```

#### 通过 Transfinite 插值生成
生成一维结构网格：
```cpp
Transfinite Curve{curveTagList} = nodeNumber <Using direction ratio>;
```
其中 `direction` 用于表示结点间距的变化方向，可以是：
```cpp
Progression  // 结点间距按几何级数 从一端向另一端 递增/递减
Bump         // 结点间距按几何级数 从两端向中间 递增/递减
```
示例：
```cpp
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Line(1) = {1, 2};
Transfinite Curve{1} = 20 Using Progression 2;
Mesh 1;
```

生成二维结构网格：
```cpp
Transfinite Surface{surfaceTagList}
<= {/* 三个或四个顶点 */pointTagList}>
<orientation>;
```
其中 `orientation` 表示三角形的取向：
```cpp
Left       // 全部向左倾斜
Right      // 全部向右倾斜
Alternate  // 倾斜方向交错变化
```
示例：
```cpp
SetFactory("OpenCASCADE");
Rectangle(1) = {0.0, 0.0, 0.0, 3.0, 2.0};
Transfinite Surface{1} = { PointsOf{ Surface{1}; } } Right;
Mesh 2;
```

生成三维结构网格（必须先在三维实体的表面生成结构网格）：
```cpp
Transfinite Volume{volumeTagList}
<= {/* 六个或八个顶点 */pointTagList}>;
```
示例：
```cpp
SetFactory("OpenCASCADE");
Box(1) = {0.0, 0.0, 0.0, 3.0, 2.0, 1.0};
Transfinite Surface{Boundary{Volume{1};}};
Recombine Surface{:};
Transfinite Volume{1};
Recombine Volume{1};
Mesh 3;
```

### 其他命令
```cpp
// 生成 dim 维网格：
Mesh dim;
// 通过分裂细化网格：
RefineMesh;
// 选择网格优化算法，algorithm 可以是 Gmsh 或 Netgen：
OptimizeMesh algorithm;
// 设置单元阶数：
SetOrder order;
// 删去冗余结点：
Coherence Mesh;
// 将网格分为 part 块：
PartitionMesh part;
```

详见《[Other geometry commands](http://gmsh.info/doc/texinfo/gmsh.html#Other-geometry-commands)》。

### 网格选项
详见《[Mesh options](http://gmsh.info/doc/texinfo/gmsh.html#Mesh-options)》。

# 文件格式

MSH 文件 (即扩展名为 `.msh` 的文件) 用于存储网格信息 (结点位置和连接关系) 以及与之关联的属性数据 (位移, 速度, 应力等).
MSH 文件有两种模式:
- ASCII 文本模式: 所有信息均以 ASCII 字符表示.
- 二进制模式: 一些信息由 ASCII 字符表示, 浮点型坐标值和整型编号以二进制表示.

一个 `MSH` 文件由以下一个或多个段落组成.
每个段落都是以一对`起始符`和`结束符`为界的 ASCII 字符或二进制数据块.
起始符有以下几种:

| 起始符 | 内容 |
| -------- | ---- |
| `$MeshFormat` | 网格格式 |
| `$PhysicalName` (optional) | 物理名 |
| `$Entities` | 实体 |
| `$PartitionedEntities` (optional) | 分块实体 |
| `$Nodes` | 结点 |
| `$Elements` | 单元 |
| `$Periodic` (optional) | 周期关系 |
| `$GhostElements` (optional) | 幽灵单元 |
| `$NodeData` (optional) | 结点数据 |
| `$ElementData` (optional) | 单元数据 |
| `$ElementNodeData` (optional) | 单元结点数据 |

起始符的 `$` 后加上 `End` 就是对应的结束符, 例如: 起始符 `$Nodes` 所对应的结束符为 `$EndNodes`.
相同类型的段落可以重复多次, 属性数据可以分属不同文件, 但要求 `Nodes` 位于 `Elements` 之前.

凡是未识别的起始符都视作开启一个注释段, 直到对应的结束符, 例如:
```cpp
$Comments
...
$EndComments
```

结点和单元的标签不要求连续.

下面以脚本文件 [`rectangle.geo`](./rectangle.geo) 生成的网格文件 [`rectangle.msh`](./rectangle.msh) 为例来说明 MSH 文件格式.

## `MeshFormat`
```cpp
$MeshFormat
4 0 8
$EndMeshFormat
```
三个数 (即使在二进制模式下, 也用字符表示) 的含义依次为:
- MSH 格式`版本`;
- 文件`模式`, `0` 为文本模式, `1` 为二进制模式;
- `字长`, 即 `sizeof(double)` 的值, 一般为 `8`.

## `PhysicalNames`
这一段用于描述物理对象信息, 以便后面按物理对象查找结点和单元.
```cpp
$PhysicalNames
3
1 1 "LeftBoundary"
1 2 "RightBoundary"
2 3 "Domain"
$EndPhysicalNames
```
各行的含义依次为:
- 第一行的 `3` 表示本文件中有 `3` 个物理实体;
- 随后三行分别为各物理实体的信息:
  - 前两个整数 (即使在二进制模式下, 也用字符表示) 分别表示`维度`和`标签`;
  - 最后的字符串 (最长可含 `127` 个字符) 为`名称`.

## `Entities`
这一段用于描述初等实体信息, 以便后面按初等实体查找结点和单元.

### 总数
第一行表示各维度初等实体的个数:
```cpp
4 4 1 0
```
- `4` 个零维初等实体, 即初等点;
- `4` 个一维初等实体, 即初等曲线;
- `1` 个二维初等实体, 即初等曲面;
- `0` 个三维初等实体, 即初等空间区域.

随后各行依次表示各初等实体的信息.

### `Point`s
前 `4` 行表示初等 `Point`s:
```cpp
1 0 0 0 0 0 0 0 
2 2 0 0 2 0 0 0 
3 2 1 0 2 1 0 0 
4 0 1 0 0 1 0 0 
```
各行含义如下:
- 第一个 `int` 表示该 `Point` 的`标签`.
- 随后三个 `double`s 表示`包围盒顶点坐标的最小值`.
- 随后三个 `double`s 表示`包围盒顶点坐标的最大值`.
- 随后一个 `unsigned long` 表示`物理标签的个数`:
  - 如果为 `0`, 则该行到此结束, 例如: 这里的所有各行.
  - 如果不为 `0`, 则给出相应个数 `int`, 表示该点所属`物理实体的标签`.

### `Curve`s
随后 `4` 行表示初等 `Curve`s:
```cpp
1 0 0 0 0 0 0 0 0 
2 2 0 0 2 1 0 1 2 2 2 -3 
3 0 1 0 2 1 0 0 2 3 -4 
4 0 0 0 0 1 0 1 1 2 4 -1 
```
各行含义如下:
- 第一个 `int` 表示该 `Curve` 的`标签`.
- 随后三个 `double`s 表示`包围盒顶点坐标的最小值`.
- 随后三个 `double`s 表示`包围盒顶点坐标的最大值`.
- 随后一个 `unsigned long` 表示`物理标签的个数`:
  - 如果为 `0`, 则进入下一项, 例如: 这里的第 `1`, `3` 行.
  - 如果不为 `0`, 则给出相应个数 `int`, 表示该 `Curve` 所属`物理实体的标签`, 例如:
    - `Curve{2}` 属于 `Physical Curve{2}`.
    - `Curve{4}` 属于 `Physical Curve{1}`.
- 随后一个 `unsigned long` 表示`边界点的个数`:
  - 如果为 `0`, 则该行到此结束, 例如这里的第 `1` 行.
  - 如果不为 `0`, 则给出相应个数 `int`, 表示该 `Curve` 所拥有的`边界点的标签`, 负标签值表示终点, 例如:
    - `Curve{2}` 有两个边界 `Point`s: `Point{2, -3}`.
    - `Curve{3}` 有两个边界 `Point`s: `Point{3, -4}`.
    - `Curve{4}` 有两个边界 `Point`s: `Point{4, -1}`.

### `Surface`s
`Surface` 段的格式与 `Curve` 段类似, 只不过把最后的`边界点`替换为`边界曲线`.

这里有 `1` 行表示初等 `Surface`:
```cpp
1 -1.8e+308 -1.8e+308 -1.8e+308 1.8e+308 1.8e+308 1.8e+308 1 3 4 1 2 3 4 
```
各行含义如下:
- 第一个 `int` 表示该 `Surface` 的`标签`.
- 随后三个 `double`s 表示`包围盒顶点坐标的最小值`, 这里是负无穷.
- 随后三个 `double`s 表示`包围盒顶点坐标的最大值`, 这里是正无穷.
- 随后一个 `unsigned long` 表示`物理标签的个数`:
  - 如果为 `0`, 则进入下一项.
  - 如果不为 `0`, 则给出相应个数 `int`, 表示该 `Surface` 所属`物理实体的标签`. 例如这里的 `Surface{1}` 属于 `Physical Surface{3}`.
- 随后一个 `unsigned long` 表示`边界曲面的个数`:
  - 如果为 `0`, 则该行到此结束, 例如这里的第 `1` 行.
  - 如果不为 `0`, 则给出相应个数 `int`, 表示该 `Surface` 所拥有的`边界曲线的标签`, 负标签值表示取向相反. 例如:
    - `Surface{1}` 拥有 `4` 条边界 `Curve`s: `Curve{1, 2, 3, 4}`.

### `Volume`s
`Volume` 段的格式与 `Surface` 段类似, 只不过把最后的`边界曲线`替换为`边界曲面`.

## `PartitionedEntities`
暂时不用.

## `Nodes`
这一段用于描述`结点位置`以及`结点与实体的归属关系`.

第一行表示各维度实体的个数:
```cpp
15 6
```
- 第一个 `unsigned long` 表示`实体总数`. 这里有 `15` 个实体:
    - `6` 个 `Point`s,
    - `7` 条 `Curve`s,
    - `2` 片 `Surface`s,
    - `0` 块 `Volume`.
- 第二个 `unsigned long` 表示`结点总数`. 这里有 `6` 个 `Node`s.

然后以每个实体为一组, 依次列出各组所拥有的结点.
例如第一行:
```cpp
5 0 0 1
```
- 第一个 `int` 表示`实体标签`.
- 第二个 `int` 表示`实体维度`.
- 第三个 `int` 表示`参数个数`.
- 最后一个 `unsigned long` 表示该实体上的`结点个数`. 这里为 `1`, 因此紧随其后的 `1` 行表示该实体上的 `1` 个结点的信息:
```cpp
1 0 0 0
```
- 第一个 `int` 表示`结点标签`.
- 紧随其后的三个 `double`s 表示`结点坐标`.

其他实体上的结点信息可以按同样的方法读出.

## `Elements`
这一段用于描述`单元类型`, `结点与单元的归属关系`以及`单元与实体的归属关系`.

第一行表示各维度实体的个数:
```cpp
5 5
```
- 第一个 `unsigned long` 表示`实体总数`. 这里有 `5` 个实体:
  - `0` 个 `Point`s,
  - `3` 条 `Curve`s,
  - `2` 片 `Surface`s,
  - `0` 块 `Volume`.
- 第二个 `unsigned long` 表示`单元总数`. 这里有 `5` 个 `Element`s:
  - `0` 个零维单元,
  - `3` 个一维单元,
  - `2` 个二维单元,
  - `0` 个三维单元.

然后以每个实体为一组, 依次列出各组所拥有的单元.
例如第一行:
```cpp
7 1 1 1
```
- 第一个 `int` 表示`实体标签`.
- 第二个 `int` 表示`实体维度`.
- 第三个 `int` 表示`单元类型`. 这里为 `1`, 表示`二结点直线单元`. 
- 最后一个 `unsigned long` 表示该实体上的`单元个数`. 这里为 `1`, 因此紧随其后的 `1` 行表示该实体上的 `1` 个单元的信息:
```cpp
7 2 3 
```
- 第一个 `int` 表示`单元标签`.
- 紧随其后的 `int`s 表示`结点标签`.
- 这一行的完整语义是: `7` 号单元有 `2` 个结点 (从单元类型推断), 结点标签分别为 `2` 和 `3`.

接下来四行表示另外两个一维单元:
```cpp
10 1 1 1
10 4 1 
11 1 1 1
13 6 5 
```
最后四行表示两个二维 (四结点四边形) 单元:
```cpp
$Elements
2 2 3 1
11 1 5 6 4 
3 2 3 1
12 5 2 3 6 
$EndElements
```

### 单元结点编号

详见《[Node ordering](http://gmsh.info/doc/texinfo/gmsh.html#Node-ordering)》。
