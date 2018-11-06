# Programming in [Python](https://docs.python.org/3/index.html)

## Quick Start

### [Introduction to Programming in Python](https://introcs.cs.princeton.edu/python/)
Robert Sedgewick 为大一新生编写的 CS 入门教程, 非常适合*零基础*的初学者.

### [The Python Tutorial](https://docs.python.org/3/tutorial/index.html)
最*官方*的教程, 可读性不如前者.

## [Built-in Data Types](https://docs.python.org/3/library/stdtypes.html)

### Numeric Types

#### `int`
Python 里的 `int` 为无限字长整数: 字长会根据需要自动扩展.
```python
print(2**31)  # 2147483648
print(2**32)  # 4294967296
print(2**63)  # 9223372036854775808
print(2**64)  # 18446744073709551616
print(2**65)  # 36893488147419103232
```

#### `float`
Python 里的 `float` *通常*为 C/C++ 里的 `double`, 即 IEEE 754 里的双精度浮点数.
```python
a = 1.0 / 3
b = 2.0 / 3
c = 3.0 / 3
print(b - a)  # 0.3333333333333333
print(c - b)  # 0.33333333333333337
print(b - a == c - b)  # False
```

#### `complex`
```python
x = 4.0
y = 3.0
z = complex(x, y)
print(z.real)  # 4.0
print(z.imag)  # 3.0
print(z.conjugate())  # (4-3j)
```

### Boolean Types
```python
print(1 + 1 == 2)  # True
print(1 + 1 == 3)  # False
```

### Sequences

#### [Common Sequence Operations](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)
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

*Immutable* 类型可以作为 `hash()` 的输入参数, 从而可以作为 `set` 和 `dict` 的关键词:

```python
p = (0.0, 0.2, -0.3)  # a tuple
print(hash(p))  # hashable
s = set()  # an unordered container
s.add(p)   # p can be used as a key
print(p)
```

*Mutable* 类型一般不可以作为 `hash()` 的输入参数, 从而也就不可以作为 `set` 和 `dict` 的关键词, 但它们支持更多与修改 (mutate) 相关的操作:

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

#### [`list`](https://docs.python.org/3/library/stdtypes.html#lists), [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuples), [`range`](https://docs.python.org/3/library/stdtypes.html#ranges)

`list` 属于 mutable 容器, `tuple` 和 `range` 属于 immutable 容器.

Python 里*一切皆引用*, *赋值 (assign)*的含义是: 将左端的*符号*与右端的*数据*进行*绑定 (bind)*. 比较下面两个赋值语句: 
```python
a = [1, 2, 3]
b = a
c = a.copy()
print(a, b, c)  # [1, 2, 3] [1, 2, 3] [1, 2, 3]
a[1] = 0
print(a, b, c)  # [1, 0, 3] [1, 0, 3] [1, 2, 3]
```
第一个没有实际拷贝数据, 第二个对数据进行了*浅*拷贝. 其中, *浅 (shallow)* 表示只拷贝元素的第一层引用:
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

`range` 只提供容器的借口, 并不需要实际存储所有数据. 例如下面的代码段, 实际存储开销只有十几个字节:

```python
for i in range(1000000):
    print(i)
```

#### [`str`](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)

`str` 类型, 又称*字符串*类型, 是一种 (通过 Unicode 编码来表示的) 字符构成的 immutable 序列.

Python 中允许使用成对的单引号 `'...'`, 或成对的双引号 `"..."`, 或成对的三重单引号 `'''...'''`来创建字符串. 其中, 单引号对最常用; 而三重单引号对中的内容可以跨行, 一般用于*文档字符串 (docstring)*.
字符串也可以通过作用在对象上的 `str()` 函数来创建: 凡是定义了 `__str__()` 方法的 Python 类型, 都可以通过作为 `str()` 函数的输入参数.

之前提到的*序列通用操作*以及 *immutable 序列所特有的操作*, 对于 `str` 类型依然适用.
除此之外, `str` 类型还有一些特殊的操作. 完整列表参见 [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods), 这里列举一些常用的操作:
- 小写转大写:
```python
'hello, world'.capitalize()  # 'Hello, world'
'hello, world'.title()       # 'Hello, World'
'hello, world'.upper()       # 'HELLO, WORLD'
```
- 将制表符 `\t` 替换为空格, 以使各列为指定宽度, 默认宽度为 `8`:
```python
'01\t012\t0123\t01234'.expandtabs( )  # '01      012     0123    01234'
'01\t012\t0123\t01234'.expandtabs(8)  # '01      012     0123    01234'
'01\t012\t0123\t01234'.expandtabs(4)  # '01  012 0123    01234'
'01\t012\t0123\t01234'.expandtabs(3)  # '01 012   0123  01234'
'01\t012\t0123\t01234'.expandtabs(2)  # '01  012 0123  01234'
```
- 构造格式化字符串, 详见 [Format String Syntax](https://docs.python.org/3/library/string.html#formatstrings):
```python
'The sum of 1 + 2 is {0}'.format(1+2)  # 'The sum of 1 + 2 is 3'
'{:<30}'.format('left aligned')  # 'left aligned                  '
'{:>30}'.format('right aligned') # '                 right aligned'
'{:,}'.format(1234567890)  # '1,234,567,890'
point = (3, 5)
'X: {0[0]};  Y: {0[1]}'.format(point)  # 'X: 3;  Y: 5'
'{0:e}'.format(314.1592653589793)    # '3.141593e+02'
'{0:f}'.format(314.1592653589793)    # '314.159265'
'{0:.4f}'.format(314.1592653589793)  # '314.1593'
'{0:g}'.format(314.1592653589793)    # '314.159'
'{0:.4g}'.format(314.1592653589793)  # '314.2'
```
- 以指定*分隔符 (delimiter)* 拆分字符串:
```python
'3.14, 0, 8'.split(', ')  # ['3.14', '0', '8']
'3.14,0,8'.split(',')     # ['3.14', '0', '8']
'3.14 0 8'.split()        # ['3.14', '0', '8']
```
- 合并字符串:
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

### Unordered Containers

#### [`set`, `frozenset`](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)
`set` 属于 mutable 容器, `frozenset` 属于 immutable 容器.

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

#### [`dict`](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)
```python
d = {"one": 1, "three": 3, "two": 2, "four": 4}
values = list(d.values())
pairs = [(v, k) for k, v in d.items()]
print(pairs)  # [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
```

## [The Python Standard Library](https://docs.python.org/3/library)

### [Data Types](https://docs.python.org/3/library/datatypes.html)

#### [`array` --- Efficient arrays of numeric values](https://docs.python.org/3/library/array.html#module-array)

`list` 和 `tuple` 都属于*异质 (heterogeneous)* 容器: 其中的元素可以是不同类型. 这种便利是通过牺牲效率而获得的. 对于*同质 (homogeneous)* 的数据, 这种效率损失可以通过利用标准库的 `array` 来避免:

```python
import array
a = array.array('i')  # create an array of type signed int, typically int32
a.append(3)
a.append(5)
print(a[0], a[-1])
```
然而, `array` 仅仅是一种容器, 并不支持加减乘除等算术运算. 如果有这类需求, 应该考虑使用 `numpy` 中的 [`ndarray`](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html).

#### [`heapq` --- Heap queue algorithm](https://docs.python.org/3/library/heapq.html)

只提供了最小二叉堆算法, 数据存储在一个 `list` 里:

```python
import heapq

heap = list()  # the actual container
heapq.heappush(heap, x)
y = heapq.heappop(heap)  # pop the top element
y = heap[0]  # access the top element without popping it
heapq.heappushpop(heap, x)  # better than a heappush() followed by a heappop()
heapq.heapreplace(heap, x)  # better than a heappop() followed by a heappush()
```
典型应用场景为 "从 $N$ 个元素的容器中找出最大的 $K$ 个数据":

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
时间复杂度为 $N \lg K$, 优于先排序再输出的 $N \lg N$.


### [Debugging and Profiling](https://docs.python.org/3/library/debug.html)

#### [The Python Profilers](https://docs.python.org/3/library/profile.html)

#### [`timeit` --- Measure execution time of small code snippets](https://docs.python.org/3/library/timeit.html)

```python
timeit.default_timer()
# The default timer, which is always time.perf_counter(), since Python 3.3.
```
典型应用场景:
```python
from timeit import default_timer as timer

start = timer()
do_something()
end = timer()
print(end - start)
```

### [Development Tools](https://docs.python.org/3/library/development.html)

#### [`unittest` --- Unit testing framework](https://docs.python.org/3/library/unittest.html)

清华大学的[软件工程](http://www.xuetangx.com/courses/course-v1:TsinghuaX+34100325_X+sp/info)公开课介绍了[单元测试](http://www.xuetangx.com/courses/course-v1:TsinghuaX+34100325_X+sp/courseware/1714014c1c1949cf84074431dc7d6a99/8623fff7bc7c4c69bced4a88620b73db/)的概念.

### [Python Runtime Services](https://docs.python.org/3/library/python.html)

#### [`abc` --- Abstract Base Classes](https://docs.python.org/3/library/abc.html)

```python
import abc
from numpy import pi

class GeometricObject2d(abc.ABC):
    @abc.abstractmethod
    def area(self):
        pass

class Circle(GeometricObject2d):  
    def __init__(self, x, y, r):
        self._x = x
        self._y = y
        self._r = r 
    def area(self):
        return pi * self._r**2

if __name__ == '__main__':
    c = Circle(0, 0, 2)
    print(c.area())
```

## Third-party Libraries

### [NumPy --- Numerical Python](https://docs.scipy.org)

最完整最准确的参考资料是 [Numpy Reference Guide](https://docs.scipy.org/doc/numpy/reference/) 和 [Numpy User Guide](https://docs.scipy.org/doc/numpy/user/). 由于需要频繁查阅, 为提高页面加载速度, 建议将 [HTML+zip](https://docs.scipy.org/doc/numpy-1.15.1/numpy-html-1.15.1.zip) 下载到本地主机, 解压后打开 `index.html` 即可进入.

[The N-dimensional array (`ndarray`)](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)

[NumPy for MATLAB users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)

### [SciPy --- Scientific Python](https://docs.scipy.org)

最完整最准确的参考资料是 [SciPy Reference Guide](https://docs.scipy.org/doc/scipy/reference/). 由于需要频繁查阅, 为提高页面加载速度, 建议下载 [HTML+zip](https://docs.scipy.org/doc/scipy-1.1.0/scipy-html-1.1.0.zip) 到本地主机, 解压后打开 `index.html` 即可进入.

### [Matplotlib --- A Python Plotting Library](https://matplotlib.org)

[Pyplot tutorial](https://matplotlib.org/tutorials/introductory/pyplot.html)

## Before Serious Programming

### [Style Guide](https://www.python.org/dev/peps/pep-0008/)

### [PyCharm](https://www.jetbrains.com/pycharm/)

- [Download](https://www.jetbrains.com/pycharm/download/)
- [Tutorials](https://confluence.jetbrains.com/display/PYH/PyCharm+Tutorials)
  - [Getting Started with PyCharm](https://confluence.jetbrains.com/display/PYH/Getting+Started+with+PyCharm)
  - [Exploring the IDE. Quick Start Guide](https://confluence.jetbrains.com/display/PYH/Exploring+the+IDE.+Quick+Start+Guide)
  - [Creating and running a Python unit test](https://confluence.jetbrains.com/display/PYH/Creating+and+running+a+Python+unit+test)