# GEO 文件说明
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

## 通用命令及选项
在这里, *通用*指的是某项功能不专属于几何/网格/求解器/后处理模块.

### 注释
注释是给人阅读的辅助信息, parser 会忽略这些内容.
GEO 文件里的注释采用 C++ 风格:

- `//` 后一行以内的内容均为注释;
- 从 `/*` 到 `*/` 之间的所有内容均为注释.

除注释以外, 所有空白字符 (空格 `' '`, 制表符 `\t`, 换行符 `\n`) 也都被 parser 忽略.

### 表达式
GEO 表达式的取值有两种类型: 字符型, 浮点型 (没有整型).
对于计算结果有确定取值类型的表达式, 可以根据计算结果的类型将其归为*浮点型表达式 (Floating Point Expressions)* 或*字符型表达式 (Character Expressions)*.
此外, 还有一种计算结果类型不定的表达式, 专门用于表示颜色信息, 因此称为*颜色表达式 (Color Expressions)*.

#### 浮点型表达式
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

#### 字符型表达式
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

#### 颜色表达式
*颜色表达式 (Colors Expressions)* 用于表示颜色信息, 可以是以下任意一种形式:
```cpp
colorName
{red, green, blue}  // [0, 255] 之间的整数, 表示红绿蓝分量数值
{red, green, blue, alpha}  // 前三个同上, 最后一个表示透明度
colorOption
```

### 运算符
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

### 内置函数
所有函数名的首字母均为大写, 除以下几个函数以 `F` 为首字母外, 其余函数均为其本名 (如 `Sin`, `Cos`, `Tan`):

| 函数 | 功能 |
| ---- | ---- |
| `Fabs(x)` | 绝对值 |
| `Fmod(x, y)` | `x % y`, 结果与 `x` 同号 |

完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [Built-in functions](http://gmsh.info/doc/texinfo/gmsh.html#Built_002din-functions).

### 自定义宏
暂时不用.

### 控制流
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

### 通用命令
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

### 通用选项
暂时不用.

## 几何模块
### CAD 内核
CAD 内核可以在 GEO 文件头部通过以下命令之一设定:
```cpp
SetFactory("built-in");
SetFactory("OpenCASCADE");
```

第一种为 Gmsh 自带的简易 CAD 内核, 只支持一些简单几何对象的创建和操作.
由于采用了*边界表示 (Boundary Representation)* 法, 所有几何实体必须通过*自底向上 (bottom-up)* 方式创建:
1. 先创建 `Point`, 
2. 再以 `Point`s 为边界或控制点创建 `Curve`, 
3. 再以 `Curve`s 为边界创建 `Surface`, 
4. 最后以 `Surface`s 为边界创建 `Volume`.

第二种为开源 CAD 系统 [OpenCASCADE](https://www.opencascade.com) 的内核, 支持一些高级几何对象的创建和操作.
如果没有特殊的需求, 建议使用这种内核.

### 创建初等实体
只具有几何意义的对象称为*初等实体 (elementary entity)*.
初等实体在创建时, 被赋予一个正整数 (*非正整数*为系统保留) 编号, [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 中称其为*标签 (tag)*.
这些 `tag` 满足:
- 每个初等 `Point` 具有唯一的 `tag` 
- 每个初等 `Curve` 具有唯一的 `tag` 
- 每个初等 `Surface` 具有唯一的 `tag` 
- 每个初等 `Volume` 具有唯一的 `tag` 

多数命令的语法与 C++ 相同, 尤其要注意: 每个语句最后的 `;` 不能省略.

- 圆括号 `()` 中的编号表示*创建*一个新的实体.
- 花括号 `{}` 中的编号表示*引用*一个已有的实体.
- 尖括号 `< >` 之间的内容为可选项.

#### `Point`s
创建三维点:
```cpp
Point(pointTag) ={
    x, y, z  // 直角坐标分量
    <, /* 单元尺寸 */elementSize>
};
```

#### `Curve`s
通过连接两点创建直线段:
```cpp
Line(curveTag) = {startPointTag, endPointTag};
```

通过一组点创建样条曲线:
```cpp
Bezier(curveTag) = {pointTagList};
BSpline(curveTag) = {pointTagList};
Spline(curveTag) = {pointTagList};
```

创建圆弧:
```cpp
// Built-in 或 OpenCASCADE 均可,
// 如果使用 Built-in 内核, 则弧度必须严格小于 Pi:
Circle(curveTag) ={
    startPointTag,
    centerPointTag,
    endPointTag
};
// 必须用 OpenCASCADE 内核:
Circle(curveTag) ={
    centerX, centerY, centerZ,
    radius<, startAngle, endAngle>
};
```

#### `Surface`s
如果使用 built-in 内核, 必须按以下流程:
```cpp
// 先创建一个或多个曲线环 (Curve Loop):
Curve Loop(curveLoopTag) = {curveTagList};
// 再创建平面区域:
Plane Surface(surfaceTag) = {curveLoopTagList};
```
其中, 第一个曲线环表示外边界, 其余曲线环表示内边界. 一个有效的曲线环必须满足:
- 封闭;
- 列表中的曲线*有序 (ordered)* 并且具有相同*取向 (oriented)*, 曲线标签前加负号表示反向.

如果使用 OpenCASCADE 内核, 可以通过以下命令快速创建平面区域:
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
如果使用 built-in 内核, 必须按以下流程:
```cpp
// 先创建一个或多个*曲面环 (Surface Loop)*:
Surface Loop(surfaceLoopTag) = {surfaceTagList};
// 再创建空间区域 (三维流形):
Volume(volumeTag) = {surfaceLoopTagList};
```
其中, 第一个曲面环表示外边界, 其余曲面环表示内边界. 一个有效的曲面环必须满足:
- 封闭;
- 列表中的曲面*有序 (ordered)* 并且具有相同*取向 (oriented)*, 曲面标签前加负号表示反向.

如果使用 OpenCASCADE 内核, 可以通过以下命令快速创建空间区域 (三维流形):
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
复合实体仍然是几何实体, 它由多个具有相同维度的初等实体组成.
在生成网格时, 这些初等实体的整体将被视作一个几何实体, 即允许一个单元跨越多个初等实体的边界.

```cpp
Compound Curve(curveTag) = {curveTagList};
Compound Surface(surfaceTag) = {surfaceTagList};
```

#### 通过拉伸创建
通过拉伸低维实体来创建高维实体:
```cpp
// 通过 平移 拉伸:
Extrude{
    vectorX, vectorY, vectorZ  // 平移向量
}{
    entityList  // 被拉伸对象
};
// 通过 旋转 拉伸:
Extrude{
    {axisX, axisY, axisZ},     // 旋转轴
    {pointX, pointY, pointZ},  // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
};
// 通过 平移 + 旋转 拉伸:
Extrude{
    {vectorX, vectorY, vectorZ},  // 平移向量
    {axisX, axisY, axisZ},        // 旋转轴
    {pointX, pointY, pointZ},     // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
};
```

### 创建物理实体
一组相同维度的初等实体可以组合成一个*物理实体 (physical entity)*, 以便赋予它们物理意义.
例如: 材料属性, 载荷分布, 边界条件等.

每个物理实体也有唯一的 `tag`, 这里的*唯一*也是针对同一维度的物理实体而言的.
除此之外, 每个物理实体还可以有一个字符串表示的名称.

```cpp
Physical Entity(tag | name<, tag>) <+|->= {entityTagList};
```
这里的 `Entity` 可以是 `Point`, `Curve`, `Surface`, `Volume` 中的任意一个.

### 编辑几何实体

#### 布尔运算
*布尔运算 (Boolean Operation)* 就是将几何区域看作点集的集合运算.
只有 OpenCASCADE 内核支持布尔运算.
所有布尔运算都是通过一条作用在两个实体列表上的指令来完成的:

```cpp
BooleanOperation{passiveEntityList}{toolEntityList}
```
- `BooleanOperation` 代表某种布尔运算, 可以是 `BooleanIntersection`, `BooleanUnion`, `BooleanDifference` 之一.
- `passiveEntityList` 代表*被动 (passive)* 实体列表, `toolEntityList` 代表*工具 (tool)* 实体列表, 它们可以是
```cpp
<Physical> Curve | Surface | Volume{tagList};  // ; 不能省略
<... | Delete;>  // 运算完成后删去对应的实体
```

新版 Gmsh 支持将运算结果存储为新的实体:
```cpp
BooleanOperation(newEntityTag) = {passiveEntityList}{toolEntityList};
```

示例:
```cpp
SetFactory("OpenCASCADE");
Rectangle(1) = {-5, -5, 0, 10, 10};
Disk(2) = {0, 0, 0, 2};
BooleanDifference(news) = {Surface{1}; Delete;}{Surface{2}; Delete;};
Mesh 2;
```
[demos/boolean](https://gitlab.onelab.info/gmsh/gmsh/tree/master/demos/boolean/) 中有更多示例.

#### 几何变换

```cpp
// 按相同比例放缩:
Dilate{
    {centerX, centerY, centerZ}, 
    factor
}{entityList};
// 按不同比例放缩:
Dilate{
    {centerX, centerY, centerZ},
    {factorX, factorY, factorZ}
}{entityList};
// 旋转:
Rotate{
    {axisX, axisY, axisZ},
    {pointX, pointY, pointZ},
    angle
}{entityList};
// 关于平面对称:
Symmetry{
    A, B, C, D  // A*x + B*y + C*z + D = 0
}{entityList};
// 平移:
Translate{
    {vectorX, vectorY, vectorZ},  // 平移向量
}{entityList};
```
其中 `entityList` 可以是
```cpp
<Physical> Point | Curve | Surface | Volume{tagList}; ...
// 或
Duplicata{<Physical> Point | Curve | Surface | Volume{tagList}; ...};
```

#### 提取边界

```cpp
// 提取边界上低一维的实体, 返回其标签:
Boundary{entityList};
// 提取边界上低一维的实体, 作为一个复合实体返回其标签:
CombinedBoundary{entityList};
// 提取边界上的点, 返回其标签:
PointsOf{entityList};
```

#### 删除

```cpp
删除坐标相同的冗余点:
Coherence;
删除列表中的实体:
<Recursive> Delete{Entity{tagList}; ...};
```
这里的 `Entity` 可以是 `Point`, `Curve`, `Surface`, `Volume` 中的任意一个.
如果列表中的某个实体被列表以外的其他实体所依赖, 则不执行 `Delete` 命令.
`Recursive` 表示 `Delete` 命令递归地作用到所有次级实体上.

### 几何选项
详见 [5.2. Geometry options](http://gmsh.info/doc/texinfo/gmsh.html#Geometry-options).

## 网格模块

### 物理实体对网格的影响
- 如果没有创建物理实体, 那么所有单元都会被写入网格文件.
- 如果创建了物理实体, 那么
    - 默认情况下, 只有物理实体上的单元会被写入网格文件, 结点和单元会被重新编号.
    - 通过设置 `Mesh.SaveAll` 选项或使用命令行参数 `-save_all` 可以保存所有单元.

### 设定单元尺寸
单元尺寸可以通过以下三种方式设定:
- 如果设定了 `Mesh.CharacteristicLengthFromPoints`, 那么可以为每个 `Point` 设定一个*特征长度 (Characteristic Length)*.
- 如果设定了 `Mesh.CharacteristicLengthFromCurvature`, 那么网格尺寸将与*曲率 (Curvature)* 相适应.
- 通过背景网格或标量场设定.
这三种方式可以配合使用, Gmsh 会选用最小的一个.

常用命令:
```cpp
// 修改点的特征长度:
Characteristic Length{pointTagList} = length;
```

完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [6.3.1. Specifying mesh element sizes](http://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes).

### 生成结构网格
Gmsh 所生成的网格都是*非结构的 (unstructured)*, 即各单元的取向和结点邻接关系完全由其结点列表决定, 而不要求相邻单元之间有其他形式的关联, 因此不能算是真正意义上的*结构 (structured)* 网格.

所有结构网格单元 (四边形, 六面体, 三棱柱) 都是通过合并单纯形 (三角形, 四面体) 而得到的:
```cpp
// 将指定曲面上的 三角形 合并为 四边形:
Recombine Surface{surfaceTagList};
// 将指定空间区域里的 四面体 合并为 六面体 或 三棱柱:
Recombine Volume{volumeTagList};
```

#### 通过拉伸生成
与几何模块中的同名函数类似, 只是多了一个 `layers` 参数:
```cpp
// 通过 平移 拉伸:
Extrude{
    vectorX, vectorY, vectorZ  // 平移向量
}{
    entityList  // 被拉伸对象
    layers
};
// 通过 旋转 拉伸:
Extrude{
    {axisX, axisY, axisZ},  // 旋转轴
    {pointX, pointY, pointZ},  // 旋转轴上任意一点
    angle
}{
    entityList  // 被拉伸对象
    layers
};
// 通过 平移 + 旋转 拉伸:
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
`layers` 用于设定拉伸方式, 有以下几种形式可以选择:
```cpp
Layers{elementNumber};  // 拉伸方向的单元数
Layers {                    // 两个列表的长度必须一致
    {elementNumberList},    // 各层的单元数
    {normalizedHeightList}  // 各层的相对高度
};
Recombine Surface{surfaceTagList};  // 将指定曲面上的 三角形 合并为 四边形
Recombine Volume{volumeTagList};    // 将指定空间区域里的 四面体 合并为 三棱柱 或 六面体
```

获取拉伸所得实体的标签:
```cpp
num[] = Extrude {0,0,1} { Surface{1}; Layers{10}; };
// num[0] 为拉伸后的顶面
// num[1] 为拉伸出的空间区域
```

#### 通过 Transfinite 插值生成
生成一维结构网格:
```cpp
Transfinite Curve{curveTagList} = nodeNumber <Using direction ratio>;
```
其中 `direction` 用于表示结点间距的变化方向, 可以是:
```cpp
Progression  // 结点间距按几何级数 从一端向另一端 递增/递减
Bump         // 结点间距按几何级数 从两端向中间 递增/递减
```
示例:
```cpp
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Line(1) = {1, 2};
Transfinite Curve{1} = 20 Using Progression 2;
Mesh 1;
```

生成二维结构网格:
```cpp
Transfinite Surface{surfaceTagList}
<= {/* 三个或四个顶点 */pointTagList}>
<orientation>;
```
其中 `orientation` 表示三角形的取向:
```cpp
Left       // 全部向左倾斜
Right      // 全部向右倾斜
Alternate  // 倾斜方向交错变化
```
示例:
```cpp
SetFactory("OpenCASCADE");
Rectangle(1) = {0.0, 0.0, 0.0, 3.0, 2.0};
Transfinite Surface{1} = {PointsOf{Surface{1};}} Right;
Mesh 2;
```

生成三维结构网格 (必须先在三维实体的表面生成结构网格):
```cpp
Transfinite Volume{volumeTagList}
<= {/* 六个或八个顶点 */pointTagList}>;
```
示例:
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
// 生成 dim 维网格:
Mesh dim;
// 通过分裂细化网格:
RefineMesh;
// 选择网格优化算法, algorithm 可以是 Gmsh 或 Netgen:
OptimizeMesh algorithm;
// 设置单元阶数:
SetOrder order;
// 删去冗余结点:
Coherence Mesh;
// 将网格分为 part 块:
PartitionMesh part;
```
完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [5.1.8 Miscellaneous](http://gmsh.info/doc/texinfo/gmsh.html#Miscellaneous-geometry-commands).

### 网格选项
详见 [6.4. Mesh options](http://gmsh.info/doc/texinfo/gmsh.html#Mesh-options).


## 求解器模块
暂时不用.

## 后处理模块
暂时不用.
