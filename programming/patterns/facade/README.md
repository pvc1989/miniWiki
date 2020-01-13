# Facade

## 意图
Provide a unified interface to a set of interfaces in a subsystem.
Facade defines a higher-level interface that makes the subsystem easier to use.

为子系统所含的接口集合提供一个单一的接口，使得子系统更易于使用。

## 用途
- 为复杂的子系统提供一个简单的接口。
- 将子系统与用户或其他子系统解耦。
- 为子系统的每一层各提供一个入口，简化各层之间的依赖关系。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **门面** (`Facade`)
  - 知道 **子系统** 所含的各个类的功能。
  - 将用户的请求转发到 **子系统** 中的某个类 (`Foo1`, `Foo2`, `Foo3`)。
- **子系统** (`SubSystem`)
  - 通过一些类 (`Foo1`, `Foo2`, `Foo3`) 实现一定的功能。
  - 处理 **门面** 对象所转发的任务。
  - 不知道 **门面** 的存在。
- **用户** (`Client`)
  - 通过 **门面** 的接口来使用子系统。
  - 不需要知道 **子系统** 含有哪些类。
