---
title: Python
---

# 入门教程

- [The Python Tutorial](https://docs.python.org/3/tutorial/index.html)
  - 最『官方』的教程
- [Introduction to Programming in Python](https://introcs.cs.princeton.edu/python/)
  - Robert Sedgewick 为大一新生编写的 CS 入门教程，非常适合『零基础』的初学者。

# [数据类型](https://docs.python.org/3/library/stdtypes.html)

## 数值类型

### `int`
Python 里的 `int` 为『无限字长整数』：字长会根据需要自动扩展。
```python
print(2**31)  # 2147483648
print(2**32)  # 4294967296
print(2**63)  # 9223372036854775808
print(2**64)  # 18446744073709551616
print(2**65)  # 36893488147419103232
```

### `float`
Python 里的 `float` 通常为 C/C++ 里的 `double`，即 IEEE 754 里的『双精度浮点数』。
```python
a = 1.0 / 3
b = 2.0 / 3
c = 3.0 / 3
print(b - a)  # 0.3333333333333333
print(c - b)  # 0.33333333333333337
print(b - a == c - b)  # False
```

### `complex`
```python
x = 4.0
y = 3.0
z = complex(x, y)
print(z.real)  # 4.0
print(z.imag)  # 3.0
print(z.conjugate())  # (4-3j)
```

## 逻辑类型
```python
print(1 + 1 == 2)  # True
print(1 + 1 == 3)  # False
```

## 无序容器

### 集合：[`set`, `frozenset`](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)

`set` 为可变容器，`frozenset` 为不可变容器，二者均支持数学中的集合操作。

```python
a = {i*2 for i in range(9)}
b = {i*3 for i in range(6)}
c = {i*6 for i in range(3)}
print(a)  # {0, 2, 4, 6, 8, 10, 12, 14, 16}
print(b)  # {0, 3, 6, 9, 12, 15}
print(c)  # {0, 12, 6}
print(a.intersection(b) == c)  # True
print(b.union(c) == b)         # True
print(b.difference(c))  # {9, 3, 15}
```

### 字典：[`dict`](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)

`dict` 是以『键-值对 (key-value pair)』 为元素的容器，并且支持按键查值。

```python
d = {"one": 1, "three": 3, "two": 2, "four": 4}
values = list(d.values())
pairs = [(v, k) for k, v in d.items()]
print(pairs)  # [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
```

### 可散列对象

Python 自带的容器 `set` 和 `dict` 都是由『散列表 (hash table)』实现的。
`set` 中的元素与 `dict` 中元素的键，都必须是『可散列对象 (hashable object)』，即可以被用作函数 `hash()` 的实参的那些对象。

## 顺序容器（序列）

### [通用序列操作](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)<a href name="common-sequence-operations"></a>

```python
x in s
x not in s
s + t  # concatenation
s * n  # adding s to itself n times
n * s  # same as above
s[i]
s[start:stop]  # s[start] included, s[stop] excluded
s[start:step:stop]
len(s)
min(s)
max(s)
s.count(x)
s.index(x[, i[, j]])  # index of the first occurrence of x in s (at or after index i and before index j)
```

『不可变对象 (immutable object)』是可散列的，因此可以被用作 `set` 中的元素或 `dict` 中元素的键：

```python
p = (0.0, 0.2, -0.3)  # a tuple
print(hash(p))  # hashable
s = set()  # an unordered container
s.add(p)   # p can be used as a key
print(p)
```

『可变对象 (mutable object)』一般不是可散列的，从而不可以被用作 `set` 中的元素或 `dict` 中元素的键，但它们支持与『改变该容器 (mutate the container)』相关的操作：

```python
s[i] = x
s[i:j] = t
del s[i:j]
s[i:j] = []  # same as above
s[i:j:k] = t
del s[i:j:k] 
s.append(x)
s.clear()
s.copy()  # creates a shallow copy of s
s[:]      # same as above
s += t
s *= n  # updates s with its contents repeated n times
s.insert(i, x)  # inserts x into s at the index given by i
s.pop([i])  # retrieves the item at i and also removes it from s
s.remove(x)  # remove the first x from s
s.reverse()
```

### 泛型序列：`list`, `tuple`, `range`

[`list`](https://docs.python.org/3/library/stdtypes.html#lists) 属于可变序列，而 [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuples) 和 [`range`](https://docs.python.org/3/library/stdtypes.html#ranges) 属于不可变序列。

Python 里一切皆『引用 (reference)』，因此『赋值 (assign)』的语义为：将左端的『符号』与右端的『数据』进行『绑定 (bind)』。
比较下面两个赋值语句：

```python
a = [1, 2, 3]
b = a
c = a.copy()
print(a, b, c)  # [1, 2, 3] [1, 2, 3] [1, 2, 3]
a[1] = 0
print(a, b, c)  # [1, 0, 3] [1, 0, 3] [1, 2, 3]
```

其中 `b = a` 没有实际拷贝数据，而 `c = a.copy()` 对数据进行了『浅拷贝 (shallow copy)』，即只拷贝元素的第一层引用。
具体效果如下：

```python
a = [[1, 2], [3, 4], [5, 6]]
b = a
c = a.copy()
print(a)  # [[1, 2], [3, 4], [5, 6]]
print(b)  # [[1, 2], [3, 4], [5, 6]]
print(c)  # [[1, 2], [3, 4], [5, 6]]
a[1][1] = 0
print(a)  # [[1, 2], [3, 0], [5, 6]]
print(b)  # [[1, 2], [3, 0], [5, 6]]
print(c)  # [[1, 2], [3, 0], [5, 6]]
```

`range` 只提供容器的接口，并不需要实际存储所有数据。因此下面的代码段实际只消耗了十几个字节的内存：

```python
for i in range(1000000):
    print(i)
```

### 字符序列：[`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)

`str` 是一种由（按 Unicode 编码的）字符构成的不可变序列。

Python 中允许使用成对的单引号 `'...'` 或成对的双引号 `"..."` 或成对的三重单引号 `'''...'''` 来创建字符串。
其中，单引号对最常用；而三重单引号对中的内容可以跨行，一般用于『文档字符串 (docstring)』。
字符串也可以通过作用在对象上的 `str()` 函数来创建：凡是定义了 `__str__()` 方法的类型，都可以通过作为 `str()` 函数的实参。

之前提到的[序列通用操作](#common-sequence-operations)，及不可变序列所特有的操作，对于 `str` 类型依然适用。
除此之外，`str` 类型还有一些特殊的操作，完整列表参见《[String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)》。
这里列举一些常用的操作：

- 小写转大写：
  ```python
  'hello, world'.capitalize()  # 'Hello, world'
  'hello, world'.title()       # 'Hello, World'
  'hello, world'.upper()       # 'HELLO, WORLD'
  ```
- 将制表符 `\t` 替换为空格，以使各列为指定宽度，默认宽度为 `8`：
  ```python
  '01\t012\t0123\t01234'.expandtabs( )  # '01      012     0123    01234'
  '01\t012\t0123\t01234'.expandtabs(8)  # '01      012     0123    01234'
  '01\t012\t0123\t01234'.expandtabs(4)  # '01  012 0123    01234'
  '01\t012\t0123\t01234'.expandtabs(3)  # '01 012   0123  01234'
  '01\t012\t0123\t01234'.expandtabs(2)  # '01  012 0123  01234'
  ```
- 构造格式化字符串，详见《[Format String Syntax](https://docs.python.org/3/library/string.html#formatstrings)》：
  ```python
  'The sum of 1 + 2 is {0}'.format(1+2)  # 'The sum of 1 + 2 is 3'
  '{:<30}'.format('left aligned')  # 'left aligned                  '
  '{:>30}'.format('right aligned') # '                 right aligned'
  '{:,}'.format(1234567890)  # '1,234,567,890'
  point = (3, 5)
  'X: {0[0]};  Y: {0[1]}'.format(point)  # 'X: 3;  Y: 5'
  'X: {0};  Y: {1}'.format(point[0], point[1])  # fortmat(x, y)
  '{0:e}'.format(314.1592653589793)    # '3.141593e+02'
  '{0:f}'.format(314.1592653589793)    # '314.159265'
  '{0:.4f}'.format(314.1592653589793)  # '314.1593'
  '{0:g}'.format(314.1592653589793)    # '314.159'
  '{0:.4g}'.format(314.1592653589793)  # '314.2'
  ```
  或 Python 3.6 起支持并推荐的 f-strings，详见《[Format String Literals](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)》：
  ```python
  x = 314.1592653589793
  f'x = {x:e}'    # 'x = 3.141593e+02'
  f'x = {x:f}'    # 'x = 314.159265'
  f'x = {x:.4f}'  # 'x = 314.1593'
  f'x = {x:g}'    # 'x = 314.159'
  f'x = {x:.4g}'  # 'x = 314.2'
  ```
- 以指定『分隔符 (delimiter)』拆分字符串：
  ```python
  '3.14, 0, 8'.split(', ')  # ['3.14', '0', '8']
  '3.14,0,8'.split(',')     # ['3.14', '0', '8']
  '3.14 0 8'.split()        # ['3.14', '0', '8']
  ```
- 合并字符串：
  ```python
  words = [str(x) for x in range(1000000)]  # a iterable container of strings
  # a trivial but slow method is via the + operator
  long_str_slow = words[0]
  for i in range(1, len(words)):
      long_str_slow += ', ' + words[i]
  # a better option is via the join() method
  long_str_fast = ', '.join(words)
  print(long_str_fast == long_str_slow)  # True
  ```

## [标准库中的数据类型](https://docs.python.org/3/library/datatypes.html)

### 小顶堆算法：[`heapq`](https://docs.python.org/3/library/heapq.html)

`heapq` 只提供了[顶部最小二叉堆『算法 (algorithm)』](https://visualgo.net/en/heap)，被操纵的数据需要存储在一个 `list` 里：

```python
import heapq

heap = list()  # the actual container
heapq.heappush(heap, x)
y = heapq.heappop(heap)  # pop the top element
y = heap[0]  # access the top element without popping it
heapq.heappushpop(heap, x)  # better than a heappush() followed by a heappop()
heapq.heapreplace(heap, x)  # better than a heappop() followed by a heappush()
```

典型应用场景：从含有 $N$ 个元素的容器中找出最大的 $K$ 个元素。代码如下：

```python
import heapq

huge_container = [-x for x in range(100000)]
K = 10
heap = list()
for x in huge_container:
    if len(heap) < K:
        heapq.heappush(heap, x)
    elif heap[0] < x:
        heapq.heapreplace(heap, x)
    else:
        pass
print(heap)  # [-9, -8, -5, -6, -7, -1, -4, 0, -3, -2]
```
时间复杂度为 $N\log(K)$，比『先排序、再输出』的 $N\log(N)$ 复杂度更优。

# 抽象机制

## 函数 (Function)
定义函数：
```python
def fib(n):
    assert n >= 0, 'n must be positive'
    n = int(n)
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
```
调用函数：
```python
assert fib(0) == 1
assert fib(1) == 1
assert fib(2) == 2
assert fib(3) == 3
assert fib(4) == 5
assert fib(5) == 8
assert fib(6) == 13
```

『函数名』可以被用作其他函数的实参。对于特别简单的函数，可以用『`lambda` 表达式』省去定义部分（故『lambda 表达式』又被称为『匿名函数』）：
```python
pairs = [(1, 'd'), (2, 'c'), (3, 'b'), (4, 'a')]
pairs.sort(key=lambda pair: pair[1])
print(pairs)  # [(4, 'a'), (3, 'b'), (2, 'c'), (1, 'd')]
```

## 类 (Class)
定义类：
```python
class Vector(object):

    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def __str__(self):
        return '(' + str(self._x) + ', ' + str(self._y) + ')' 

    def dot(self, vector):
        assert isinstance(vector, Vector)
        return self._x * vector._x + self._y * vector._y
```
使用类：
```python
v = Vector(x=1.0/3, y=1.0 - 2.0/3)
print(v)  # (0.3333333333333333, 0.33333333333333337)
print(v.dot(v))  # 0.22222222222222224
```

## 模块 (Module)
我们可以把一组常量、函数、类的定义放在一个『脚本 (script)』文件里，这样的文件称为『模块 (module)』。
一个模块中的函数或类可以被其他模块『引入 (import)』，这样就在各模块之间的就形成了一个『体系 (hierarchy)』，其中最顶层的模块称为『主模块 (main module)』。
模块提供了一种『命名空间 (namespace)』管理机制，可以有效地避免来自不同模块的同名函数、同名类的冲突。

创建一个 `fibonacci.py` 文件：
```python
def fib(n):
    assert n >= 0, 'n must be positive'
    n = int(n)
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    print(fib(n))
```

模块最主要的功能是向其他模块提供函数、类的定义，有三种引入方式：
- 将模块名 `fibonacci` 引入到当前命名空间中：
  ```python
  import fibonacci
  fibonacci.fib(10)  # 89
  ```
- 将模块 `fibonacci` 中的函数名 `fib` 引入到当前命名空间中：
  ```python
  from fibonacci import fib
  fib(20)  # 10946
  ```
- 将模块 `fibonacci` 中的`所有`公共函数/类名引入到当前命名空间中：
  ```python
  from fibonacci import *
  fib(30)  # 1346269
  ```

其中，第一种方式引起名称冲突的可能性最小，对命名空间的保护最好；第三种对命名空间的保护最差，要尽量避免。

模块还可以像普通脚本一样被『执行 (executing)』，通常用于进行简单的测试。
一种好的习惯是将可执行代码置于文件末尾，例如：
```python
if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    print(fib(n))
```
这样，就可以在终端中这样来执行这个脚本：
``` shell
python3 fibonacci.py 20
```

## 包 (Package)
包是比模块更高一层的封装。
一个含有 `__init__.py` 的目录就是一个包，而该目录下的其他 `.py` 文件就是这个包的子模块。

Python 社区有许多开源的包或模块。
最常用的获取方式是从 [Python Package Index (PyPI)](https://pypi.org) 安装（`python3` 对应 `pip3`）：
```shell
pip --help  # 获取 pip 帮助
pip list  # 列出已安装的包
pip install --help  # 安装 pip install 帮助
pip install numpy  # 安装 numpy
pip install --upgrade numpy  # 更新 numpy
pip install --user numpy    # 为当前用户安装 numpy, 是 Debian 系统下的默认方式
pip install --system numpy  # 为所有用户安装 numpy, 系统管理员使用
```

# 数值计算

## [`numpy`, `scipy`](https://www.scipy.org) for 数值计算

最完整最准确的参考资料是
- [Numpy Reference Guide](https://docs.scipy.org/doc/numpy/reference/) 
- [Numpy User Guide](https://docs.scipy.org/doc/numpy/user/)
- [SciPy Reference Guide](https://docs.scipy.org/doc/scipy/reference/)

由于需要频繁查阅, 建议将 HTML+zip ([`numpy`](https://docs.scipy.org/doc/numpy-1.15.1/numpy-html-1.15.1.zip), [`scipy`](https://docs.scipy.org/doc/scipy-1.1.0/scipy-html-1.1.0.zip)) 到本地主机，解压后打开 `index.html` 即可进入。

[The N-dimensional array (`ndarray`)](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)

[NumPy for MATLAB users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)

## [`matplotlib`](https://matplotlib.org) for 数据可视化

[Pyplot tutorial](https://matplotlib.org/tutorials/introductory/pyplot.html)

# 文件读写
## [`io`](https://docs.python.org/3/library/io.html) for 数据流读写
在 Python 中，文件被抽象成数据『流 (stream)』，所有文件『读写 (IO)』操作都是在流对象上进行的。
除此之外，还有一种『内存中的流 (in-memory stream)』，它支持所有流上的抽象操作，但不与某个文件绑定。

本质上，所有信息在计算机中都是以二进制形式表示的，因此『二进制读写 (Binary IO)』适用于所有类型的文件。
但对文本文件而言，直接使用二进制读写并不是一种好的方案，因为字符的二进制表示依赖于编码 (ASCII, UTF-8, Unicode)，一些特殊字符（换行符）还与操作系统相关。
为了隔离这些细节，Python 标准库提供了专门的『文本读写 (Text IO)』接口。

以指定模式打开或创建一个文件，将其绑定到一个流对象上：
```python
f = open('filename.txt', mode='r')  # read-only (default)
f = open('filename.txt', mode='w')  # write-only
f = open('filename.txt', mode='a')  # append
```
所有被打开的文件都应当被妥善关闭：
```python
f.close()
```
一种简洁且安全的写法是把 `open()` 置于 `with` 之后：
```python
with open('filename.txt') as f:
    read_data = f.read()
print(f.closed)  # True
```

从当前位置开始，读取指定个数的字符，返回一个 `str` 对象：
```python
s = f.read(4)  # read 4 chars
s = f.read()   # read all
```
从当前位置开始，读取一行（到当前行末尾或文件末尾为止），返回一个 `str` 对象：
```python
s = f.readline()  # '\n' included
```
将字符串写入流：
```python
f.write('hello, world\n')
```

## [`sys.stdin`, `sys.stdout`, `sys.stderr`](https://docs.python.org/3/library/sys.html#sys.stdin) for 标准读写
从『标准输入 (`stdin`)』中逐行读取信息并进行处理：
```python
import sys
for line in sys.stdin:
    # Process each line, which is an str object.
```

## [`stdio`](https://introcs.cs.princeton.edu/python/) from *Introduction to Programming in Python*

前一节介绍的文本读写操作要求输入输出对象都是 `str` 类型，因此在读写 `int` 或 `float` 数据时，需要频繁地进行类型转换。

《[Introduction to Programming in Python](https://introcs.cs.princeton.edu/python/)》引入了一个 [`stdio`](https://introcs.cs.princeton.edu/python/code/stdio.py.html) 模块，对『标准输入输出流 (`stdio`)』上常用数据类型的读写操作进行了封装。
《[1.5  Input and Output](https://introcs.cs.princeton.edu/python/15inout)》详细介绍了该模块的用法。

要使用该模块，必须先下载 [`stdio.py`](https://introcs.cs.princeton.edu/python/code/stdio.py) 文件，然后将其置于 Python 搜索路径中（例如『当前工作目录』下）。

将任意类型的数据写到『标准输出流 (`stdout`)』：
```python
import stdio

x = 3.1415926
stdio.write(x)  # write '3.1415926' to stdout
stdio.writeln(x)  # write '3.1415926\n' to stdout
stdio.writef("%6.2f", x)  # write '  3.14' to stdout
```

从『标准输入流 (`stdin`)』读取指定类型的数据：
```python
# 查询 stdin 中是否还有待处理数据：
stdio.isEmpty()
stdio.hasNextLine()
# 读取下一个数据（以空白字符为间隔），按指定类型返回：
stdio.readInt()
stdio.readFloat()
stdio.readBool()
stdio.readString()
stdio.readLine()
# 读取 stdin 中所有待处理数据，按指定类型返回：
stdio.readAll()
stdio.readAllInts()
stdio.readAllFloats()
stdio.readAllBools()
stdio.readAllStrings()
stdio.readAllLines()
```

结合操作系统提供的数据流『重定向 (redirect)』及『管道 (pipe)』功能，可以将该模块应用到一般文本文件上：
```shell
# redirecting:
python3 randomseq.py 1000 > data.txt
python3 average.py < data.txt
# piping:
python3 randomseq.py 1000 | python3 average.py
```
这两个脚本的功能分别为:
- [`randomseq.py`](https://introcs.cs.princeton.edu/python/15inout/randomseq.py)：生成给定数量的随机数，输出为序列。
- [`average.py`](https://introcs.cs.princeton.edu/python/15inout/average.py)：计算给定序列的平均值。

# 图形界面

## [`PyQt5`](https://pypi.org/project/PyQt5/)

- [Learn PyQt](https://www.learnpyqt.com/)
  - [15-minute-apps](https://github.com/learnpyqt/15-minute-apps)
  - [Getting started with PyQt5](https://www.learnpyqt.com/courses/start/)
  - [Creating applications with Qt Designer](https://www.learnpyqt.com/courses/qt-creator/)

# 软件开发

## 代码规范：[PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257)

## [`argparse`](https://docs.python.org/3/library/argparse.html) for 命令行解析

```python
import argparse

parser = argparse.ArgumentParser(
    prog = 'ProgramName',
    description = 'What the program does',
    epilog = 'Text at the bottom of help')

parser.add_argument('-n', '--n_element',  # option that takes a value
    default=10, type=int, help='number of elements')
parser.add_argument('-v', '--verbose',    # on/off flag
                    action='store_true')

args = parser.parse_args()
print(args)
```

## [`profile`](https://docs.python.org/3/library/profile.html) for 函数调用分析

## [`timeit`](https://docs.python.org/3/library/timeit.html) for 测量运行时间

```python
timeit.default_timer()
# The default timer, which is always time.perf_counter(), since Python 3.3.
```
典型应用场景：
```python
from timeit import default_timer as timer

start = timer()
do_something()
end = timer()
print(end - start)
```

## [`unittest`](https://docs.python.org/3/library/unittest.html) for 单元测试

『单元测试 (unit test)』是『测试驱动开发 (Test Driven Development, TDD)』的基础。

典型用法：
```python
import unittest

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
```

## [`abc`](https://docs.python.org/3/library/abc.html) for 定义抽象基类（接口）

```python
import abc
from numpy import pi

class GeometricObject2d(abc.ABC):
    @abc.abstractmethod
    def area(self):
        pass

class Circle(GeometricObject2d):  
    def __init__(self, x: float, y: float, r: float):
        self._x = x
        self._y = y
        self._r = r
    def area(self):
        return pi * self._r**2

class Square(GeometricObject2d):  
    def __init__(self, x: float, y: float, a: float):
        self._x = x
        self._y = y
        self._a = a
    def area(self, scale: float):
        # The signiture is different from that of the abstract method,
        # but still passes the instantiation check.
        return self._a * self._a * scale

if __name__ == '__main__':
    c = Circle(0, 0, 2)
    assert c.area() == 2 * 2 * pi
    s = Square(0, 0, 2)
    assert s.area(1) == 2 * 2 * 1
    assert s.area(3) == 2 * 2 * 3
```

抽象基类（接口）的『正确』用法，可参见《[设计原则](./principles/README.md)》及《[设计模式](./patterns/README.md)》。

