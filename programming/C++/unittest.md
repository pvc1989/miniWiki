# 单元测试

好的测试代码应当满足:
- 每个`功能点`的测试应当是`独立的` [independent] 和 `可重复的` [repeatable].
- `测试代码`应当被妥善组织, 以反映`被测试代码`的结构.
- 测试代码应当是`可移植的` [portable] 和 `可复用的` [reusable].
- 测试代码编译或运行失败时, 测试系统应当给出恰好能反映关键问题的信息.
- 测试应当能够`快速`编译和运行.

## Google Test
[Google Test](https://github.com/google/googletest) 是一个开源的 C++ 测试`框架` [framework], 主要用来做单元测试, 也可以进行其他测试 (回归测试, 集成测试, 压力测试等).

### 文档
必读:
- [Primer](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) --- 入门教程
- [Generic Build Instructions](https://github.com/google/googletest/blob/master/googletest/README.md) --- 构建指南

选读:
- [Samples](https://github.com/google/googletest/blob/master/googletest/docs/samples.md)
- [Advanced Topics](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md)
- [FAQ](https://github.com/google/googletest/blob/master/googletest/docs/faq.md)

### 术语

由于历史的原因, Google Test 所采用的术语与其他测试框架或文献所采用的通用术语略有区别:

| 名称     | 通用术语   | Google 术语 (旧) | 含义                     |
| ------- | --------- | --------------- | ----------------------- |
| 断言    | Assertion  | Assertion       | 程序正确运行时应当成立的条件 |
| 测试函数 | Test Case  | Test (Function) | 由一组断言组成的单个测试用例 |
| 测试集   | Test Suite | Test Case       | 由多个测试用例所组成的测试集 |

目前 (2019/03), Google Test 正在进行一轮较大规模的重构, 新的版本将使用与通用术语一致的 API. 因此, 建议在新编写的测试代码中使用 `TestSuite` 取代 `TestCase`.

### 断言 (Assertion)
`断言`是所有测试的基础.
在 Google Test 中, 断言是通过`宏` [macro] 机制实现的, 形式上与`函数` [function] 类似.
每一种断言都有 `ASSERT_*` 和 `EXPECT_*` 两个版本:

|         |    `ASSERT_*`    |     `EXPECT_*`      |
| :-----: | :--------------: | :-----------------: |
| **程度** | 致命的 [fatal] | 非致命的 [nonfatal] |
| **行为** | 跳出当前`测试函数` |   跳出当前`断言语句`    |
| **后果** |     资源泄露     |    影响后续结果     |
| **建议** |     谨慎使用     |      推荐使用       |

基础断言有以下两种 (只写 `EXPECT` 版本), 原则上可以用它们表达任何断言:

| 断言形式                   | 成立条件                     |
| -------------------------- | ---------------------------- |
| `EXPECT_TRUE(condition);`  | `bool(condition)` 为 `true`  |
| `EXPECT_FALSE(condition);` | `bool(condition)` 为 `false` |

#### 通用比较

常用的二元比较断言有以下六种 (只写 `EXPECT` 版本), 其中 `a` 表示`实际` [actual] 值, `e` 表示`期望` [expected] 值:

| 断言形式           | 成立条件 | 缩写含义                 |
| ------------------ | -------- | ------------------------ |
| `EXPECT_EQ(a, e);` | `a == e` | EQual to                 |
| `EXPECT_NE(a, e);` | `a != e` | Not Equal to             |
| `EXPECT_LT(a, e);` | `a < e`  | Less Than                |
| `EXPECT_LE(a, e);` | `a <= e` | Less than or Equal to    |
| `EXPECT_GT(a, e);` | `a > e`  | Greater Than             |
| `EXPECT_GE(a, e);` | `a >= e` | Greater than or Equal to |

#### 字符串比较
如果两个字符串中有一个是 `std::string` 对象, 则应当使用 `EXPECT_EQ` 或 `EXPECT_NE`:
```cpp
#include <string>
auto std_string = std::string("hello, world");
auto std_nullstr = std::string();
EXPECT_NE(std_string, std_nullstr);
```
如果两个字符串中都是 C-风格字符串, 则应当使用 `EXPECT_STREQ` 或 `EXPECT_STRNE`.
如果用的是 `EXPECT_EQ` 或 `EXPECT_NE`, 则实际进行比较的是两个`地址`:
:

```cpp
auto c_string = "hello, world";
auto c_nullstr = "";
EXPECT_STRNE(c_string, c_nullstr);  // 比较字符串内容
EXPECT_NE(c_string, c_nullstr);     // 比较地址
```

#### 浮点数比较
两个浮点数由不同 (业务逻辑的) 程序生成的浮点数`几乎不可能`严格相等.
因此用 `EXPECT_EQ` 进行比较通常是不合适的.
Google Test 为此专门设计了用于比较浮点数的断言 (只写 `EXPECT` 版本):

| 断言形式                   | 成立条件                     |
| ------------------------- | ---------------------------- |
| `EXPECT_FLOAT_EQ(a, e);`  | 两个 `float` 型浮点数几乎相等  |
| `EXPECT_DOUBLE_EQ(a, e);` | 两个 `double` 型浮点数几乎相等  |
| `EXPECT_NEAR(a, b, eps);` | `abs(a - b) < eps` 为 `true` |

#### 出错信息
Google Test 本身会在断言出错时显示一些预制的信息.
测试设计者可以用 `<<` 运算符为断言补充出错信息:
```cpp
ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";

for (int i = 0; i < x.size(); ++i) {
  EXPECT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;
}
```

### 测试函数和测试集
#### 创建普通测试
创建一个`测试函数`:
```cpp
TEST(TestSuiteName, TestName) {
  // 测试内容
}
```
其中, 函数名为 `TEST`, 返回类型为空, `TestSuiteName` 和 `TestName` 分别是`测试集`和`测试函数`的标识符 (必须是合法的 C++ 标识符, 并且不含有下划线).

```cpp
  // Returns the factorial of n.
int Factorial(int n) {
  return (n == 1) ? 1 : n * Factorial(n-1);
}

// Tests factorial of 0.
TEST(FactorialTest, HandlesZeroInput) {
  EXPECT_EQ(Factorial(0), 1);
}

// Tests factorial of positive numbers.
TEST(FactorialTest, HandlesPositiveInput) {
  EXPECT_EQ(Factorial(1), 1);
  EXPECT_EQ(Factorial(2), 2);
  EXPECT_EQ(Factorial(3), 6);
  EXPECT_EQ(Factorial(8), 40320);
}
```

#### 创建 Fixture
在测试一个类 (例如 `Foo`) 时, 不同的测试函数往往会在一组相同的 `Foo` 对象上进行测试.
为了提高代码复用率, 可以将它们封装进一个 `::testing::Test` 的派生类 (称作 `FooTest`) 中:
1. 在 `FooTest` 内部以 `protected:` 为默认访问控制级别.
2. 将需要被重复使用的数据定义为 `FooTest` 的成员.
3. 如果需要申请动态资源并重设数据, 则为 `FooTest` 定义`默认构造函数`或重写 `SetUp()` 方法.
4. 如果需要释放动态资源, 则为 `FooTest` 定义`析构函数`或重写 `TearDown()` 方法.
5. 如果需要, 定义其他方法.
6. 定义测试函数时, 用 `TEST_F()` 代替 `TEST()`, 在测试函数内部可以直接使用 `FooTest` 的成员.

假设有如下的待测试类:
```cpp
template <typename E>  // E is the element type.
class Queue {
 public:
  Queue();
  void Enqueue(const E& element);
  std::unique_ptr<E> Dequeue();  // Returns nullptr if the queue is empty.
  std::size_t size() const;
};
```
为其创建 fixture:
```cpp
class QueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
     q1_.Enqueue(1);
     q2_.Enqueue(2);
     q2_.Enqueue(3);
  }

  void TearDown() override { }  // 不需要释放动态资源, 可以省略

  Queue<int> q0_;
  Queue<int> q1_;
  Queue<int> q2_;
};
```
创建测试集:
```cpp
TEST_F(QueueTest, IsEmptyInitially) {
  EXPECT_EQ(q0_.size(), 0);
}

TEST_F(QueueTest, DequeueWorks) {
  auto uptr = q0_.Dequeue();
  EXPECT_EQ(uptr, nullptr);
  uptr.release();

  uptr.reset(q1_.Dequeue().release());
  ASSERT_NE(uptr, nullptr);
  EXPECT_EQ(*uptr, 1);
  EXPECT_EQ(q1_.size(), 0);
  uptr.release();

  uptr.reset(q2_.Dequeue().release());
  ASSERT_NE(uptr, nullptr);
  EXPECT_EQ(*uptr, 2);
  EXPECT_EQ(q2_.size(), 1);
}
```

#### 运行全部测试
在 `main()` 中调用 `RUN_ALL_TESTS()`, 则运行时会执行所有被链接进可执行文件的测试.
如果所有测试全部通过, 则 `RUN_ALL_TESTS()` 返回 `0`, 否则返回 `1`.
- `RUN_ALL_TESTS()` 的返回值应当通过 `main()` 返回给操作系统.
- `RUN_ALL_TESTS()` 只应被 `main()` 调用一次.
- 必须在 `RUN_ALL_TESTS()` 前调用`::testing::InitGoogleTest()` 以便处理命令行参数.

```cpp
#include "gtest/gtest.h"

// 测试集

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

### 构建
#### 获取源代码
首先, 将托管在 [GitHub](https://github.com/google/googletest) 上的源代码仓库克隆到本地:
```shell
git clone https://github.com/google/googletest.git
```
克隆成功后, 在当前目录 (运行 `git clone` 时所在的目录) 下将得到一个 `googletest` 目录, 其结构大致如下:
```shell
googletest
├── README.md
├── CMakeLists.txt
├── googlemock
│   ├── CMakeLists.txt
│   ├── docs
│   ├── include
│   ├── src
│   ├── test
│   ├── ...
├── googletest
│   ├── CMakeLists.txt
│   ├── docs
│   ├── include
│   ├── samples
│   ├── src
│   ├── test
│   └── ...
├── ...
```
其中的 `CMakeLists.txt` 可以用来驱动 [CMake](../make.md#CMake) --- 这是目前最简单最通用的自动构建方式.

#### 构建为独立的库
假设源文件目录 (必须含有 `CMakeLists.txt`) 为 `source-dir`, 构建产物存放在 `build-dir`, 则构建过程如下:
```shell
# 生成本地构建文件, 例如 Makefile:
cmake [options] -S source-dir -B build-dir
# 调用本地构建工具, 例如 make:
cmake --build build-dir
```
其中, `[opitions]` 是可选项 (使用时不写 `[]`), 用于设置 (或覆盖 `CMakeLists.txt` 中设置过的) CMake 变量的值.
一般形式为 `-D var=value` (注意 `=` 两边没有空格), 常用的有 (可以组合使用):

| `var` (大小写敏感)               | `value` (大小写敏感) | 含义                                |
| -------------------------------- | -------------------- | ----------------------------------- |
| `CMAKE_CXX_FLAGS`                | `-std=c++11`         | 启用 C++11 标准                     |
| `gtest_build_samples`            | `ON`                 | 构建 `googletest/samples/` 中的示例 |
| `gtest_build_tests`              | `ON`                 | 构建 `googletest/tests/` 中的测试   |
| `GTEST_CREATE_SHARED_LIBRARY`    | `1`                  | 生成动态链接库                      |
| `GTEST_LINKED_AS_SHARED_LIBRARY` | `1`                  | 使用动态链接库                      |

在[图形用户界面的 CMake](https://cmake.org/download/) 里, 这些 CMake 变量可以分组显式, 查找和修改起来非常方便.

#### 集成到本地项目

|         方式         | 难度 |  路径  |  源代码  |      目标码      | 更新方式 |
| :-----------------: | :--: | :---: | :-----: | :-------------: | :-----: |
| 源代码复制进本地项目 | 容易 | 无依赖 | 独立副本 |     独立副本     |  纯手动  |
|   构建为公共静态库   | 中等 | 被依赖 | 只需一份 | 可执行文件中重复 |  半自动  |
|   构建为公共动态库   | 中等 | 被依赖 | 只需一份 |     只需一份     |  半自动  |
|  作为子项目参与构建  | 困难 | 无依赖 | 独立副本 |     独立副本     |  全自动  |

推荐采用最后一种方式, 主要包括以下两个步骤:
1. 创建 `CMakeLists.txt.in` 文件, 设置 Google Test 仓库地址, 本地源文件目录和构建目录.
2. 在本地项目的 `CMakeLists.txt` 文件中添加命令, 构建 Google Test 和本地测试.

官方文档的 [Incorporating Into An Existing CMake Project](https://github.com/google/googletest/tree/master/googletest#incorporating-into-an-existing-cmake-project) 一节给出了这两个文件的模板.
这里给出一个简单的 C++ 项目示例, [源文件目录](./unittest/googletest/)结构如下, 用 [CMake](../make.md#CMake) 来组织构建:
```
googletest
├── CMakeLists.txt
├── CMakeLists.txt.in
├── include
│   └── math.h
├── src
│   ├── CMakeLists.txt
│   └── math.cpp
└── test
    ├── CMakeLists.txt
    └── test_math.cpp
```
