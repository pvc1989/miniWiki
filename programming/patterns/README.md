# Design Patterns

## Creational Patterns

### [Abstract Factory](./AbstractFactory/README.md)

### [Builder](./Builder/README.md)

### [Factory Method](./FactoryMethod/README.md)

### [Prototype](./Prototype/README.md)

### [Singleton](./Singleton/README.md)

## Structural Patterns

### [Adapter](./Adapter/README.md)

### [Bridge](./Bridge/README.md)

### [Composite](./Composite/README.md)

### [Decorator](./Decorator/README.md)

### [Facade](./Facade/README.md)

### [Flyweight](./Flyweight/README.md)

### [Proxy](./Proxy/README.md)

## Behavioral Patterns

### 责任链 (Chain of Responsibility) 模式

> Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.
>
> 解耦请求的发送者与接收者: 将请求的 (潜在) 接收者组织成链, 并沿之传递请求, 直到请求被某个接收者处理.

### 命令 (Command) 模式

> Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
>
> 将请求封装为对象, 以便 (1) 以不同请求对客户端进行参数化, (2) 将请求编入队列或记录请求日志, (3) 支持可撤销的操作.

### 解释器 (Interpreter) 模式

> Given a language, define a represention for its grammar along with an interpreter that uses the representation to interpret sentences in the language.
>
> 为给定的语言定义一种表示方法和相应的解释器, 用以解释该语言的语句.

### 迭代器 (Iterator) 模式

> Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
>
> 提供一种遍历容器的方法, 而不暴露其内部表示.

### 仲裁者 (Mediator) 模式

> Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently.
>
> 定义一个 (仲裁者) 对象来封装一组对象的交互方式, 通过避免这组对象之间的显式引用来促进松耦合, 以便独立改变其交互行为.

### 备忘录 (Memento) 模式

> Without violating encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later.
>
> 在不破坏封装的情况下, 捕获并外化一个对象的内部状态, 使得该对象在将来能够被恢复到该状态.

### 观察者 (Observer) 模式

> Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
>
> 定义多个 <观察者> 与一个 <被观察者> 的依赖关系, 以便 <被观察者> 的状态发生改变时, 所有 <观察者> 自动获得通知并更新.

### 状态 (State) 模式

> Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.
>
> 允许一个对象根据其内部状态的变化改变行为, 使其看上去像是变换了类型.

### 策略 (Strategy) 模式

> Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
>
> 定义一组算法, 对每一个进行封装, 并使他们可以相互替换.
> 该模式使得算法和使用它的客户端可以独立变化.

### 模板方法 (Template Method) 模式

> Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.
>
> 定义算法的框架, 将某些步骤的实现延迟到派生类中.
> 该模式使得派生类可以在不改变算法的结构情况下重写某些步骤.

### 访客 (Visitor) 模式

> Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.
>
> 将作用于某个对象成员的操作表示为一个 (访客) 对象, 以便在不改变这些成员类型的情况下定义新的操作.

## Language Support

### Java

#### `implements` an `interface`

```java
public interface Comparable<T> {
  public abstract int compareTo(T that);
}

public class Point implements Comparable<Point> {
  private final double x;
  private final double y;
  
  public Point(double x, double y) {
    this.x = x;
    this.y = y;
  }
  
  @Override
  public int compareTo(Point that) {
    if (this.x < that.x) return -1;
    if (this.x > that.x) return +1;
    if (this.y < that.y) return -1;
    if (this.y > that.y) return +1;
    return 0;
  }
}
```

#### `extends` a `class`

所有用 `class` 定义的类都是 [`java.lang.Object`](https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html) 的 <子类::subclass> (或 <派生类::derived classs>).

```java
public class Object {
  public String toString();
}

public class Point extends Object {
  private final double x;
  private final double y;
  
  public Point(double x, double y) {
    this.x = x;
    this.y = y;
  }
  
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append('(');
    sb.append(this.x);
    sb.append(',').append(' ');
    sb.append(this.y);
    sb.append(')');
    return sb.toString();
  }
  
  // extended methods
  // ...
}
```

### C++

#### Inheritance

```c++
#include <iostream>
#include <string>

class Object {
 public:
  virtual std::string to_string() = 0;
};

class Point : public Object {
 private:
  const double _x;
  const double _y;
  
 public:
  Point(double x, double y) : _x(x), _y(y) { }
  virtual std::string to_string() override {
    return '(' + std::to_string(_x) + ',' + ' ' + std::to_string(_y) + ')';
  }
};

int main() {
  Point p = Point(1.0, 0.0);
  std::cout << p.to_string() << std::endl;
}
```

#### Inheritance + `template`

```cpp
#include <iostream>

template<class T>
class Comparable {
  virtual int compareTo(const T& that) = 0;
};

class Point : public Comparable<Point> {
 private:
  const double _x;
  const double _y;
  
 public:
  Point(double x, double y) : _x(x), _y(y) { }
  virtual int compareTo(const Point& that) override {
    if (_x < that._x) return -1;
    if (_x > that._x) return +1;
    if (_y < that._y) return -1;
    if (_y > that._y) return +1;
    return 0;
  }
};

int main() {
  using P = Point;
  P o = P(0.0, 0.0);
  P p = P(1.0, 0.0);
  std::cout << p.compareTo(o) << std::endl;
}
```
