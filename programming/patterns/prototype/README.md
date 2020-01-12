# Prototype

## 意图
Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

通过给定一个原型实例来指定待创建对象的类型，并通过复制该原型来创建新的对象。

## 用途
- 让系统独立于产品的创建、组合、表示。
- 允许在运行期指定待创建对象的类型。
- 避免工厂类数量的膨胀（例如：定义一个平行于产品家族的工厂家族）。
- 通过安装原型来实现不同状态的组合。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **抽象原型** (`Prototype`)
  - 声明一个用于复制对象的接口 (`Clone()`)。
- **具体原型** (`ConcretePrototype`)
  - 实现 **抽象原型** 中的复制接口。
- **用户** (`Client`)
  - 通过调用原型对象的复制接口 (`Clone()`) 来创建新的对象。
