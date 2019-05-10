# Abstract Factory (Kit)

## 意图
Provide an interface for creating families of related or dependent objects without specifying their concrete classes.


## [类图](./Class.txt)
![](./Class.svg)

- 接口层
  - `抽象产品X` --- 定义第 `X` 种产品的接口，这里的 `X` 可以是 `A` 或 `B`。
  - `抽象工厂` --- 为 `抽象产品X` 提供一个创建方法 `createX()`。
  - `Client` --- 仅使用 `抽象工厂` 和 `抽象产品X` 的接口。
- 实现层
  - `具体产品Xi` --- 第 `X` 种抽象产品的第 `i` 种具体实现，这里的 `i` 可以是 `1` 或 `2`。
  - `具体工厂AiBj` --- 实现 `抽象工厂` 的接口，可以显式地为所有可能的 `具体产品` 组合逐一给出定义，也可以在构造函数（或模板参数）中进行参数化设置。

⚠️ `抽象产品X` 的实现方式 `i` 可以自由扩展，而种类 `X` 不易扩展。

## [Java 示例](./Demo.java)

## [C++ 示例](./Demo.cpp)
