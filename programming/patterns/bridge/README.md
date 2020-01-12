# Bridge (Handle/Body)

## 意图
Decouple an abstraction from its implementation so that the two can vary independently.

将抽象与实现解耦，以便二者独立变化。

## 用途
- 避免将抽象永久地绑定到它的某个实现，从而支持在运行期从多种实现中进行选择或切换。
- 允许通过继承对抽象和实现进行扩展。
- 对实现的修改不会影响到抽象（迫使其重新编译）。
- （通过指向实现的指针）将实现细节完全隐藏。
- 在多个对象之间共享同一个实现，并且对用户隐藏该细节。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **抽象类** (`Abstraction`)
  - 声明用户所使用的抽象接口 (`Request()`)。
  - 维护一个指向某个 **实现类** 实例的引用（或指针）。
- **细化的抽象** (`RefinedAbstraction`)
  - 扩展 **抽象类** 的接口。
- **抽象实现类** (`Implementor`)
  - 声明 **具体实现类** 的公共接口 (`RequestImpl()`)，不必与 **抽象类** 所声明的接口 (`Request()`) 一致。
- **具体实现类** (`ConcreteImplementorA`, `ConcreteImplementorB`)
  - 给出 **抽象实现类** 的具体实现。
