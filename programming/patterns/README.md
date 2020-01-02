# Design Patterns

## Creational Patterns

### [Abstract Factory](./AbstractFactory/README.md)
![](./AbstractFactory/Class.svg)

### [Builder](./Builder/README.md)
![](./Builder/Class.svg)

### [Factory Method](./factory_method/README.md)
![](./factory_method/class.svg)

### [Prototype](./Prototype/README.md)
![](./Prototype/Class.svg)

### [Singleton](./Singleton/README.md)
![](./Singleton/Class.svg)

## Structural Patterns

### [Adapter](./Adapter/README.md)
![](./Adapter/Class.svg)

### [Bridge](./Bridge/README.md)
![](./Bridge/Class.svg)

### [Composite](./Composite/README.md)
![](./Composite/Class.svg)

### [Decorator](./Decorator/README.md)
![](./Decorator/Class.svg)

### [Facade](./Facade/README.md)
![](./Facade/Class.svg)

### [Flyweight](./Flyweight/README.md)
![](./Flyweight/Class.svg)

### [Proxy](./Proxy/README.md)
![](./Proxy/Class.svg)

## Behavioral Patterns

### [Chain of Responsibility](./ChainOfResponsibility/README.md)
![](./ChainOfResponsibility/Class.svg)

### [Command](./Command/README.md)
![](./Command/Class.svg)

### [Interpreter](./Interpreter/README.md)
![](./Interpreter/Class.svg)

### [Iterator](./Iterator/README.md)
![](./Iterator/Class.svg)

### [Mediator](./Mediator/README.md)
![](./Mediator/Class.svg)

### [Memento](./Memento/README.md)
![](./Memento/Class.svg)

### [Observer](./observer/README.md)
![](./observer/class.svg)

### [State](./State/README.md)
![](./State/Class.svg)

### [Strategy](./Strategy/README.md)
![](./Strategy/Class.svg)

### [Template Method](./TemplateMethod/README.md)
![](./TemplateMethod/Class.svg)

### [Visitor](./Visitor/README.md)
![](./Visitor/Class.svg)

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
