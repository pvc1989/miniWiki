# Singleton

## 意图
Ensure a class only has one instance, and provide a global point of access to it.

确保某个类只有一个实例，并为该实例提供全局访问点。

## 用途
- 某个类只应有一个实例，该实例可以通过给定的接口被用户从任意位置访问。
- 可以通过继承来扩展，用户不需要修改已有代码就能使用扩展了的实例。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **单例** (`NonDerivableSingleton`, `BaseSingleton`, `DerivedSingleton`)
  - 将该类型的唯一实例定义为静态成员 (`instance`)。
  - 声明一个用于访问该实例的静态方法 (`GetInstance()`)。
  - 将构造函数声明为 *不可访问的*。
  - 可能需要负责创建该类型的唯一实例。
