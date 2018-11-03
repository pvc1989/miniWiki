# GNU Octave 入门

[Octave](https://www.gnu.org/software/octave/) 是一款开源数值计算软件: 在满足 [GPL](https://www.gnu.org/software/octave/license.html) 条款的前提下, 任何人都可以自由地获取并重新发布其源代码.

Octave 也可以看做是一种类似于 [MATLAB](https://www.mathworks.com) 的动态语言: 它的基础语法和许多库函数与 MATLAB 兼容, 因而在一定程度上可以将其作为后者的替代品.

## 安装

最符合开源精神的安装方式是从源代码编译. 如果想跳过编译选项设置及依赖性检查, 可以从[这里](https://www.gnu.org/software/octave/#install)下载各种操作系统下已经编译好的二进制安装文件.

## 运行

### CLI

在操作系统的 Shell 中输入 `octave`, 即可进入 Octave 命令窗口, 使用方式与 MATLAB CLI 非常类似:
```octave
A = eye(3);
b = ones(3,1);
x = A \ b;
A*x - b
```
行末不加 `';'` 则下一行输出该表达式的计算结果:
```octave
ans =

   0
   0
   0
```

CLI 常用命令:

| 命令 | 功能 |
| ---- | ---- |
| `clc` | 清屏 |
| `clear` | 清除当前工作空间 |
| `exit` 或 `quit` | 退出 Octave |

### GUI

Octave GUI 的结构及各部分的名称与 MATLAB GUI 几乎完全一样.

## 获取帮助

学习和使用 Octave 最可靠最完整的参考文档是 GNU Octave 手册, 可以在线查阅 [HTML](https://octave.org/doc/interpreter/index.html) 或下载 [PDF](https://www.gnu.org/software/octave/octave.pdf). 

如果已经在本地主机安装了 Octave 软件, 可以通过在 Octave 命令提示符后输入 `doc` 打开 GNU Octave 手册. 如果要查询某一条命令 (例如 `rand`) 的帮助, 可以在 Octave 命令提示符后输入以下两种命令之一:

```octave
help rand
doc rand
```

区别是:
- `help` 会将所有信息输出, 并立即回到 Octave 命令窗口.
- `doc` 会停留在帮助文档中, 以便上下翻阅或查找关键词, 直到按下 `Q` 才回到 Octave 命令窗口.

## 数据类型

### Scalar

| 类型 | 实现方式 | `Bytes` | 最小值 | 最大值 |
| ---- | ---- | ------ | ------ | ------ |
| `logical` | `logical` | `1` |  |  |
| `char` | `char8` | `1` | `0` | `255` |
| `int` | `int32` | `4` | `intmin` | `intmax` |
| `double` | `double` | `8` | `realmin` | `realmax` |
| `complex` | `pair<double,double>` | `16` |  |  |

#### 逻辑型

表示逻辑判断的结果, 只有 `true` 和 `false` 两个值. 尽管只需要 `1 bit`, 但仍占据 `1 byte`.

#### 字符型

用 ASCII 码表示字符, 可以看做短整型.

#### 整型

Octave 中的整数, 既非数学中的整数, 也不完全等价于 C/C++ 中的 `int32`. 以下**反常**的结果可以说明这一问题:

```octave
a = intmin
a - 1 == a
a * 2 == a
```

#### 浮点型

Octave 中的实数, 不是数学中的实数, 但完全等价于 C/C++ 中的 ` double`, 即 [IEEE 754 双精度浮点数](https://en.wikipedia.org/wiki/IEEE_754).
有几个特殊实数需要特别注意:

| 符号 | 含义 | 来源 |
| ---- | ---- | ---- |
| `Inf` 或 `inf` | 无穷大 | 分母为零, 分子不为零 |
| `NaN` | **N**ot **a** **N**umber | 不定式, 如 `0/0` |
| `NA` | **N**ot **A**vailable | `NA` 命令 |

Octave 中的复数表示为一对实数, 可以通过在整数或实数常量后紧跟 `i` 或 `j` 来创建, 或利用 `complex` 函数来构造.

```octave
c = 0 + 1i
d = 0 + 1j
c*c - d^2  % == 0
```

### Matrix

Octave 中的矩阵是一种同构 (元素类型相同) 的多维数组, 支持矩阵的加法和乘法运算, 能够动态地扩展或删减.

#### 创建

手动生成矩阵:
```octave
% ',' 或 ' ' 表示间隔, 可以省略
% ';' 表示换行, 不可省略
a = [1,2; 3,4]
b = [1 2; 3 4]
a - b
```

自动生成等距数组:
```octave
% start : step : stop
a = 0 : 10
b = 0 : 0.1 : 1
```

自动生成特殊矩阵 (若只给一个输入参数, 则输出为方阵):

| 函数 | 功能 |
| ---- | ---- |
| `eye(m, n)` | 主对角线元素为 `1.0`, 其余元素为 `0.0` |
| `ones(m, n)` | 所有元素为 `1.0` |
| `rand(m, n)` | 元素为取自 `(0.0, 1.0)` 的随机数 |
| `zeros(m, n)` | 所有元素为 `0.0` |

#### 索引

```octave
a = magic(3)
a(2, 2)  % 指定行列的元素
a(2, :)  % 第二行所有元素
a(:, 2)  % 第二列所有元素
% 当做一维数组访问:
a(4:6)  % column-major
a(:)
```

在大多数语言中, 访问数组的越界成员是非法的, 但在 Octave 中规定如下:
- 对数组的越界成员进行**赋值**会导致数组隐式地自动向外扩张.
- 对数组的越界成员进行**取值**会导致报错.

```octave
a = [1 2 3 4]
a(8) = 8  % 自动扩张
a(10)     % 报错
```

#### 转置

```octave
a = complex(magic(3), ones(3))
a.'  % 转置
a'   % 共轭转置
```

#### 拼接

```octave
a = [1,2; 3,4]
b = [5 6; 7 8]
c = [a, b]
d = [a; b]
```

### String

字符串可以看做是由字符元素构成的数组.

### Cell Array

异构数组.

### Structure

类似于 C/C++ 中的 `struct`.

### Map

类似于 Java 中的 `Map` 或 Python 中的 `dict`.

### 常用类型函数

获取对象的类型信息:

| 函数 | 功能 |
| ---- | ---- |
| `class(a)` | 返回 `a` 的类型 |
| `isa(a, "type")` | 判断 `a` 是否为 `"type"` 所表示的类型 |
| `isnumeric(a)` | 判断 `a` 是否为数值型 (整型, 浮点型) |
| `ismatrix(a)` | 判断 `a` 是否为矩阵 (二维数组) |
| `ishermitian(a)` | 判断 `a` 是否为共轭对称矩阵 |

以 `is` 开头的判断函数被称作**谓词 (predicate)**, 完整列表 (个别函数在 MATLAB 中没有定义) 参见 GNU Octave 手册的 [4.8 Predicates for Numeric Objects](https://octave.org/doc/interpreter/Predicates-for-Numeric-Objects.html#Predicates-for-Numeric-Objects).

获取对象的大小信息:

| 函数 | 功能 |
| ---- | ---- |
| `length(a)` | 第一维有多少个元素 |
| `ndims(a)` | **n**umber of **dim**ension**s** |
| `numel(a)` | **num**ber of **el**ements |
| `size(a)` | 几行几列 |
| `sizeof(a)` | 多少字节 |

## 表达式

### 运算符优先级

`*` 的优先级高于 `+`, 所以 `a + b * c` 应解释为 `a + (b * c)`.

> 运算符优先级混淆会导致程序的语法正确而算法逻辑出错. 当不确定时, 一律主动加上括号, 这样可以从根本上避免此类错误.

在 Octave 中, 各运算符 (自增自减运算符和复合赋值运算符在 MATLAB 中没有定义) 的优先级从高到低 (同组优先级相同) 依次为:

| 优先级 | 运算符 | 示例 |
| ------ | ------ | ---- |
| 1 | 函数调用 | `f(x)` |
| 1 | 数组访问 | `a(1)` |
| 1 | 异构数组访问 | `a{1}` |
| 1 | 结构体成员访问 | `point.x` |
|  |  |  |
| 2 | 后置自增, 后置自减 | `a++`, `a--` |
|  |  |  |
| 3 | 矩阵转置 | `a'`, `a.'` |
| 3 | 矩阵幂 | `a^2`, `a**2` |
| 3 | 矩阵成员幂 | `a.^2`, `a.**2` |
|  |  |  |
| 4 | 一元加, 一元减 | `+a`, `-a` |
| 4 | 前置自增, 前置自减 | `++a`, `--a` |
| 4 | 逻辑取反 | `~a`, `!a` |
|  |  |  |
| 5 | 矩阵乘 | `a * b`, `a .* b` |
| 5 | 矩阵左除 | `a / b`, `a ./ b` |
| 5 | 矩阵右除 | `a \ b`, `a .\ b` |
|  |  |  |
| 6 | 二元加, 二元减 | `+a`, `-a` |
|  |  |  |
| 7 | 序列生成 | `1 : 10` |
|  |  |  |
| 8 | 矩阵成员与 | `a & b` |
|  |  |  |
| 9 | 矩阵成员或 | `a | b` |
|  |  |  |
| 10 | 逻辑与 | `a && b` |
|  |  |  |
| 11 | 逻辑或 | `a || b` |
|  |  |  |
| 12 | 赋值, 复合赋值 | `a = 2`, `a += 2` |

## 控制流

所有控制语句都以关键词 (如 `if`, `switch`, `while`, `for`) 开头, 以 `end` 结尾. 在 Octave 中, `end` 可以加上关键词, 如 `endif`, `endswitch` 等, 但这样的代码在 MATLAB 中将会报错.

### 条件

```octave
x = rand();
if x > 0.5
  disp('large');
else
  disp('small');
end
```

```octave
x = rand();
if x > 0.67
  disp('large');
elseif x > 0.33  % 不能写成 else if
  disp('medium');
else
  disp('small');
end
```

```octave
x = randi(10)
switch x
  case 1
    disp('x == 1');
  case 2
    disp('x == 2');
  case {3, 4, 5}
    disp('x == 3 or 4 or 5');
  otherwise
    disp('x > 5');
end
```
其中任何一个 `case` 后面的代码被执行完后, 控制流直接跳转到 `end`, 这一点与 C/C++ 中的 `switch` 不同.

### 循环

寻找矩阵中的最大成员:
```octave
a = rand(1, 5)
x = a(1);
n = numel(a);
for k = 1 : n
  if a(k) > x
    x = a(k);
  end
end
disp(sprintf('max(a) == %f', x));
```
或
```octave
a = rand(1, 5)
x = a(1);
k = 1;
while k <= numel(a)
  if a(k) > x
    x = a(k);
  end
  k = k + 1;
end
disp(sprintf('max(a) == %f', x));
```

有两种方式可以跳过最内层循环的剩余部分:
- `break` 语句用于跳出最内层循环.
- `continue` 语句用于跳转到最内层循环的下一次迭代.

## 函数

### 定义函数

函数定义的基本形式为:
```octave
function [return_list] = name(input_list)
% algorithms
end
```
其中输入和输出列表可以有零个或多个参数, 例如:
```octave
function y = f()
% ...
end

function g(x, y)
% ...
end

function [u, v] = h(x, y)
% ...
end
```

### 函数文件

将函数定义放置在与之同名的文本文件中, 扩展名为 `.m`, 就创建了一个函数文件.
一个函数文件中有且仅有一个主函数, 但可以定义多个子函数:
```octave
% 文件名为 f.m
function f()
  disp('in f, calling g');
  g()
end
function g()
  disp('in g, calling h');
  h()
end
function h()
  disp('in h');
end
```
其中 `f` 为文件外部可见的主函数, `g` 和 `h` 为文件外部不可见但内部可见的子函数.

要让函数文件所定义的函数能够被 Octave 调用, 必须将该函数文件放置在 Octave 的搜索路径中. 可以通过 `addpath` 函数来添加搜索路径:
```octave
addpath("~/Desktop")
```

### 脚本文件

**脚本文件 (script)** 与函数文件类似, 也是含有一系列 Octave 命令的 `.m` 文件, 但没有用关键词 `function` 与 `end` 对代码进行封装. 脚本文件中的变量位于脚本调用语句所在的作用域中.

### 运行时间及调用信息

假设在当前工作目录下创建了一个 `fib.m` 文件, 其中定义了一个用递归的方法计算 Fibonacci 数的函数 `fib(n)`
```octave
function result = fib(n)
  n = round(n);
  assert(n >= 0)
  switch n
    case {0, 1}
      result = 1;
    otherwise
      result = fib(n-1) + fib(n-2);
  end
end
```

利用 `tic`-`toc`, 可以测量函数运行时间:
```octave
tic; fib(20); t = toc
```

利用 `profile`, 可以更精细地测量函数运行时间及调用次数等信息:
```octave
% Octave 与 MATLAB 均可用
profile clear;
profile on;
fib(20);
profile off;
p = profile('info');
```
查看测量结果的命令为:
```octave
profshow(p, 10);  % 仅 Octave 可用
profile viewer;   % 仅 MATLAB 可用
```

### 参数传递与作用域

Octave 中, 函数参数的默认传递方式是**按值传递 (pass by value)**, 即被传递的是**值**, 而不是**变量 (的地址)**:

```octave
foo = "bar";  % 变量 foo 的值为 "bar"
f(foo)        % 被传递的是 "bar" 这个值, 而不是 foo 这个变量
```

按值传递是 Octave 所保证的**语义 (semantics)**, 但该语义不一定是通过创建局部副本 (复制) 来实现的. 如果函数体中没有修改传入参数的值, 则不必进行复制, 例如:

```octave
% a_unchanged.m
function 
end
% a_changed.m
function a_changed(a)
  a(1, 1) = 1;
end
% Command Window
clear
a = rand(1000); % 8000000 Bytes
profile clear
profile on
for i = 1 : 100
  a_unchanged(a);  % no copy, very fast
  a_changed(a);    % do copy, very slow
end
profile off
p = profile('info');
profshow(p, 10);  % 仅 Octave 可用
profile viewer;   % 仅 MATLAB 可用
```

按值传递语义的一个重要推论是, 函数体内无法直接修改外部变量的值:

```octave
% f.m
function f(a)
  a(1) = 0;  % 与外部的 a 不是同一个对象
  a
end
% Command Window
a = [1 1]
f(a)  % 显示 [0 1]
a     % 显示 [1 1]
```

这里有两个名为 `a` 的变量, 它们都是**局部 (local)** 变量 (与**全局 (global)** 变量对应), 但是具有不同的**作用域 (scope)**. 作用域是一种数据保护机制, 可以保证外部环境不被局部变量污染. 在 Octave GUI 的**工作空间 (workspace)** 窗口中可以查看当前作用域中的局部变量.

> 全局变量可以绕过作用域规则, 但几乎总是意味着糟糕的设计. 应当避免使用全局变量.

### 函数作为参数

#### 函数句柄

类似于函数 C 中的函数指针, 通过 `@` + 函数名:
```octave
f = @sin
f(pi)             % 像函数一样调用
quad(f, 0, pi/2)  % 像变量一样传递
```

#### 匿名函数

类似于函数 Python 中的 lambda 表达式, 例如:
```octave
quad(     @cos  , 0, pi/2)  % 传入函数句柄
quad(@(x) cos(x), 0, pi/2)  % 传入匿名函数
```
二者运行结果相同, 但匿名函数效率很低.

## 画图

### 二维线图

| 函数 | 功能 |
| ---- | ---- |
| `plot(x, y)` | 二维点及连线 |
| `semilogx(x, y)` | 同上, 但 `x` 按对数缩放 |
| `loglog(x, y)` | 同上, 但 `x` 和`y` 都按对数缩放 |
| `contour(x, y, z)` | 二维等值线图 |
| `contourf(x, y, z)` | 同上, 但有填充 |

```octave
x = -3 : 0.2 : 3;
y = x;
[X, Y] = meshgrid(x, y);
Z = peaks(X, Y);
contour(X, Y, Z);
```

### 三维线图

| 函数 | 功能 |
| ---- | ---- |
| `plot3(x, y, x)` | 三维点及连线 |
| `contour3(x, y, z)` | 三维等值线图 |
| `mesh(x, y, z)` | 网格 |
| `surf(x, y, z)` | 曲面 |

### 标注

| 函数 | 功能 |
| ---- | ---- |
| `title` | 标题 |
| `xlabel` | 坐标轴 |
| `legend` | 图例 |

```octave
x = 0 : 0.1 : 10;
hold
plot(x, sin(x));
plot(x, cos(x));
title('Trigonometric Functions');
xlabel('x');
ylabel('y');
legend('sin(x)', 'cos(x)');
```

### 简单动画

| 函数 | 功能 |
| ---- | ---- |
| `clf` | 清屏 |
| `pause(0.5)` | 暂停 `0.5` 秒 |

```octave
x = [0, 2, 2, 0, 0];
y = [0, 0, 2, 2, 0];
cx = 1;
cy = 1;
dt = 0.02
for t = 0 : dt : 2
  clf
  hold on
  axis equal
  axis off
  plot(x, y);
  a = 8 * t;
  r = 0.5 * t;
  plot(cx + r*cos(a), cy + r*sin(a), 'b*');
  pause(dt);
end
```

## 算例

### 分子碰撞的直接数值模拟

#### 物理背景

在宏观尺度下, 气体可以视为连续介质; 但在微观尺度下, 气体应当看作是由许多分子组成的系统, 这些分子无时无刻不在做无规则热运动. 描述气体分子热运动规律的理论称为**气体动理学理论 ([The Kinetic Theory of Gases](http://www.feynmanlectures.caltech.edu/I_39.html))**, 其二维简化版本可以概括为以下几点:
- 每个分子都用一个刚性圆盘来表示: 圆盘的半径和质量分别代表分子的半径和质量, 忽略转动惯量.
- 分子与分子之间以及分子与壁面之间至多可以相切, 不允许相交.
- 碰撞是分子与分子之间以及分子与壁面之间唯一的相互作用形式.
- 所有碰撞均为瞬时刚性碰撞: 动量守恒, 角动量守恒, 能量守恒, 忽略碰撞过程所消耗的时间.

尽管该模型引入了很多简化假设, 但仍然保留了气体分子运动规律最本质的特征. 从该模型出发, 运用统计学工具, 可以导出许多宏观物理量的定义, 从而过渡到连续介质力学的范畴.

#### 计算任务

考虑一个二维区间 `0 < x < 1, 0 < y < 1`, 四周为固壁边界. 给定 `n` 个分子的半径 `r(i)` 和质量 `m(i)`, 以及 `t = 0` 时刻的位置 `x(i), y(i)` 和速度 `u(i), v(i)`. 计算 `t > 0` 时所有分子的位置和速度, 并用动画 (每隔 `0.03` 秒画一帧) 实时显示计算结果, 直到按下 `[Ctrl] + C`.
