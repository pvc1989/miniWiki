# Decorator (Wrapper)

## 意图
Attach additional responsibilities to an object dynamically.
Decorators provide a flexible alternative to subclassing for extending functionality.

动态地为一个对象添加职责。
该模式为扩展现有类的功能提供了一种比继承更灵活的替代方案。

## 用途
- 动态且透明（不影响其他对象）地为一个对象添加职责。
- 支持撤回操作。
- 用于继承不可行的场景。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **部件** (`Component`)
  - 声明可动态添加职责的对象的接口 (`Request()`)。
- **具体部件** (`ConcreteComponent`)
  - 实现 **部件** 的接口。
- **修饰器** (`Decorator`)
  - 维护一个指向 **部件** 实例的引用或指针，并提供与 **部件** 相同的接口 (`Request()`)。
- **具体修饰器** (`StateDecorator`, `BehaviorDecorator`)
  - 为 **部件** 实例添加职责，既可以添加状态 (`addedState`)，又可以添加行为 (`AddedOperation`)。
