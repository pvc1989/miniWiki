# Template Method

## 意图
Define the skeleton of an algorithm in an operation, deferring some steps to subclasses.
Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.

将算法框架定义在一个操作中，将其中用到的部分基本操作交给子类去实现。
该模式允许子类在不改变算法结构的情况下重新定义某些基本操作。

## 用途
- 算法不变的部分只需要在模板方法中实现一次，可变的部分留给子类去实现。
- 从子类中抽取公共部分，以减少代码重复。
- 通过定义一些被称为 **钩子 (hook)** 的模板方法，限制子类的扩展。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **抽象类** (`AbstractClass`)
  - 声明一些 **基本操作** (`PrimitiveOperation1()`, `PrimitiveOperation2()`)，留给 **具体类** 去实现。
  - 基于 **基本操作** 将算法骨架实现为一个 **模板方法** (`TemplateMethod()`)。
- **具体类** (`ConcreteClass`)
  - 实现 **抽象类** 中声明的 **基本操作**。
