# Strategy (Policy)

## 意图
Define a family of algorithms, encapsulate each one, and make them interchangeable.
Strategy lets the algorithm vary independently from clients that use it.

定义一组算法，分别进行封装，使它们可以互换。
该模式使得算法可以独立于它的使用者发生变化。

## 用途
- 从多种（算法、行为）方案中任选一种。
- 支持某个算法的多个变种，以体现不同的空间/时间取舍。
- 隔离使用者与算法所使用的数据结构。
- 避免冗长的条件分支。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **策略** (`Strategy`)
  - 声明算法的公共接口 (`DoSomething()`)，供 **用户** 使用。
- **具体策略** (`ConcreteStrategy`)
  - 实现 **策略** 的接口。
- **用户** (`Client`)
  - 接收 **策略** 实例 (`SetStrategy()`) 作为配置参数。
  - 维护指向 **策略** 实例的引用或指针 (`strategy`)。
  - 可以定义一个供 **策略** 访问其数据的接口 (`GetData()`)。
