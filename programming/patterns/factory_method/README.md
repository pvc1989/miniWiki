# Factory Method (Virtual Constructor)

## 意图
Define an interface for creating an object, but let subclasses decide which class to instantiate.

为创建某种产品的对象定以一个接口，由子类确定创建哪种具体产品。

## 用途
- 一个类无法预知待创建产品的具体类型。
- 一个类希望由其子类来决定待创建产品的具体类型。

## [类图](./class.txt)
![](./class.svg)

## 参与者
- **抽象产品** (`AbstractProduct`)
  - 为 **工厂方法** 所创建的对象声明接口。
- **具体产品** (`ConcreteProduct`)
  - 实现 **抽象产品** 的接口。
- **抽象创建者** (`AbstractCreator`)
  - 声明 **工厂方法** (`Create()`)，以 **抽象产品**（的指针）为返回类型。
  - **工厂方法** 可以是抽象方法，也可以给出默认实现。
  - 其他方法 (`OtherMethod()`) 可以通过调用 **工厂方法** 来创建 **抽象产品**。
- **具体创建者** (`ConcreteCreator`)
  - 实现 **抽象创建者** 中的 **工厂方法**，以 **具体产品**（的指针）为返回类型。
