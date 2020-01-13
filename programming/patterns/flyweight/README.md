# Flyweight

## 意图
Use sharing to support large numbers of fine-grained objects efficiently.

通过共享来支持高效使用大量细粒度对象。

## 用途
该模式适用于以下 **所有** 条件都满足的情形：
- 应用程序需要使用大量对象。
- 直接存储这些对象的空间开销巨大。
- 这些对象的多数状态是 **外在状态 (Extrinsic State)**。
- 许多对象可以被只含有 **内在状态 (Intrinsic State)** 的共享对象替换。
- 应用程序不依赖于这些对象的身份（地址）。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **享元/轻量级对象** (`Flyweight`)
  - 声明用于接收和响应外部状态的接口 (`Request(ExtrinsicState)`)。
- **具体享元** (`SharedFlyweight`, `UnsharedFlyweight`)
  - 实现 **享元** 的接口。
  - 所有 **享元** 都要存储其内在状态 (`intrinsicState`)，以便实现共享。
  - **享元** 的接口只是用来支持共享，但不要求所有 **具体享元** 都实现共享。非共享的版本 (`UnsharedFlyweight`) 还要存储对象的外在状态 (`extrinsicState`)。
- **享元工厂** (`FlyweightFactory`)
  - 创建并管理 **享元** 对象。
  - 确保 **享元** 对象被正确地共享。
- **用户** (`Clinet`)
  - 维护指向 **享元** 对象的引用或指针。
  - 计算或存储 **享元** 对象的外部状态。
