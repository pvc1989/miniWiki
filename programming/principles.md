# 设计原则

面向对象设计的五个基本原则（首字母串接起来构成 **SOLID**）：
- [SRP --- Single Resposibility Principle](#单一责任原则)
- [OCP --- Open--Closed Principle](#开闭原则)
- [LSP --- Liskov Substitution Principle](#Liskov-替换原则)
- [ISP --- Interface Segregation Principle](#接口分离原则)
- [DIP --- Dependency Inversion Principle](#依赖倒置原则)

## 单一责任原则

> A class should have only one reason to change.
>
> 一个类应该只有一个引起变化的原因。

## 开闭原则

> Software entities (classes, modules, functions, etc.) should be open for extension, but closed for modification.
>
> 软件实体（类、模块、函数等）应该对扩展开放，对修改封闭。

## Liskov 替换原则

> Subtypes must be substitutable for their base types.
>
> 派生类必须能替换其基类。

## 接口分离原则

> Clients should not be forced to depend on methods that they do not use.
>
> 客户端不应被强制依赖于自己用不到的方法。

## 依赖倒置原则

> High-level modules should not depend on low-level modules. Both should depend on abstractions.
>
> 高层模块不应依赖于低层模块，它们都应依赖于抽象。
>
> Abstractions should not depend on details. Details should depend on abstractions.
>
> 抽象不应依赖于细节，细节应依赖于抽象。
