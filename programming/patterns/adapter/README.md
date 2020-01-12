# Adapter (Wrapper)

## 意图
Convert the interface of a class into another interface clients expect.

将一个类的接口转换为用户所期待的另一种接口。

## 用途
- 封装一个现有类，将其接口转换为用户所期待的形式。
- 为不相关的或不可预见的类提供一个抽象接口，使其可以通过该抽象接口与用户互动（用户不依赖于第三方类，只依赖于抽象接口）。
- 对象适配器可以在无法同时继承多个现有子类的情况下，通过对象组合来适配父类的接口。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **目标** (`Target`)
  - 声明用户所期待的接口 (`Request()`)。
- **用户** (`Client`)
  - 通过 `Target::Request()` 与实现了该接口的对象互动。
- **被适配者** (`Adaptee`)
  - 提供一个已经存在、但不是用户所期待的接口 (`SpecificRequest()`)。
- **适配器** (`ObjectAdapter`, `ClassAdapter`)
  - 利用 `Adaptee::SpecificRequest()` 实现 `Target::Request()`。
