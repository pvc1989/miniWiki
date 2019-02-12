# 模板函数
`T` 的实际类型将根据 `compare` 的`静态`调用方式在`编译期`决定:

```cpp
template <typename T>  // 模板形参列表
int compare(const T& v1, const T& v2) {
  if (v1 < v2) return -1;
  if (v2 < v1) return 1;
  return 0;
}
```

`inline`, `constexpr` 等修饰符应当位于`模板形参列表`与`返回值类型`之间:

```cpp
template <typename T>
inline T min(const T&, const T&);
```

编写模板函数应当
- 尽量减少对模板形参的要求.
- 用`指向常量的引用`作为函数形参类型,  这样可以使得代码也适用于不可拷贝的类型.
- 只用 `<` 进行比较操作, 这样用于实例化模板的类型实参只需要支持 `<` 而不必支持 `>`.

## 函数重载
编译器首先根据`类型转换次数`对所有待选 (模板和非模板) 函数进行排序:
- 如果正好有一个函数的匹配程度比其他函数更高, 则它将被选中.
- 如果匹配程度最高的函数有多个, 则
  - 如果其中只有一个`非模板`, 则它将被选中.
  - 如果其中没有非模板, 且有一个`模板`比其他的更加`特殊化 (specialized)`, 则它将被选中.
  - 否则, 该调用有歧义, 编译时会报错.

# 模板类

```cpp
template <typename T> class Blob {
 public:
  typedef T value_type;
  typedef typename std::vector<T>::size_type size_type;
  Blob();
  Blob(std::initializer_list<T> il);
  size_type size() const { return data->size(); }
  bool empty() const { return data->empty(); }
  void push_back(const T& t) { data->push_back(t); }
  void push_back(T&& t) { data->emplace_back(std::move(t)); }
  void pop_back();
  T& back();
  T& operator[](size_type i);
 private:
  std::shared_ptr<std::vector<T>> data;
  // throws msg if data[i] is not valid:
  void check(size_type i, const std::string& msg) const;
};
```

## 定义成员函数
模板类的成员函数可以在类的内部或外部定义.
在内部定义的成员函数是隐式内联的.

### 在外部定义成员函数
在模板类外部定义的成员函数以 `template` 关键词 + 模板类的形参列表 开始, 例如:
```cpp
template <typename T>
void Blob<T>::check(size_type i, const std::string& msg) const {
  if (i >= data->size())
    throw std::out_of_range(msg);
}
template <typename T>
T& Blob<T>::back() {
  check(0, "back on empty Blob");
  return data->back();
}
template <typename T>
T& Blob<T>::operator[](size_type i) {
  check(i, "subscript out of range");
  return (*data)[i];
}
template <typename T>
void Blob<T>::pop_back() {
  check(0, "pop_back on empty Blob");
  data->pop_back();
}
```

### 在外部定义构造函数
```cpp
template <typename T>
Blob<T>::Blob() : data(std::make_shared<std::vector<T>>()) {}
template <typename T>
Blob<T>::Blob(std::initializer_list<T> il)
    : data(std::make_shared<std::vector<T>>(il)) {}
```
为使用第二个构造函数, 初始化列表的元素类型必须与模板类型实参兼容:
```cpp
Blob<string> articles = {"a", "an", "the"};
```

## 定义静态成员
`Foo` 的每个`实例化 (instantiation)` 都有其自己的静态成员 (数据或方法) `实例 (instance)`:
```cpp
template <typename T> class Foo {
 public:
  static std::size_t count() { return ctr; }
 private:
  static std::size_t ctr;
};
```
而每个`静态数据成员`应当有且仅有一个`定义`. 因此, 模板类的`静态数据成员`应当像`成员函数`一样, 在类的外部给出唯一的定义:
```cpp
template <typename T>
size_t Foo<T>::ctr = 0;  // 定义并初始化 ctr
```

## 使用模板类名

### 在内部使用类名
`模板类名 (name of a class template)`  (不带模板实参) 不是一种`类型名 (name of a type)`, 但在模板类自己的作用域内, 可以不受此限制:

```cpp
template <typename T> class BlobPtr {
 public:
  BlobPtr() : curr(0) {}
  BlobPtr(Blob<T>& a, size_t sz = 0) : wptr(a.data), curr(sz) {}
  T& operator*() const {
    auto p = check(curr, "dereference past end");
    return (*p)[curr];  // (*p) is the vector to which this object points
  }
  // 返回值类型写为 BlobPtr& 而不是 BlobPtr<T>&
  BlobPtr& operator++();
  BlobPtr& operator--();
 private:
  // check returns a shared_ptr to the vector if the check succeeds 
  std::shared_ptr<std::vector<T>> check(std::size_t, const std::string&) const;
  // store a weak_ptr, which means the underlying vector might be destroyed 
  std::weak_ptr<std::vector<T>> wptr;
  std::size_t curr;  // current position within the array
};
```
在这里, 自增自减运算符的返回值类型可以写为 `BlobPtr&` 而不是 `BlobPtr<T>&`. 这是因为在模板类作用域内, 编译器将`模板类名`视为带有模板实参的`类型名`:
```cpp
// 相当于
BlobPtr<T>& operator++();
BlobPtr<T>& operator--();
```

### 在外部引用类名
在模板类外部定义成员时, 模板类的作用域起始于 (带模板实参的) 类名. 因此在 `::` 之前需要显式写出模板实参, 而在其之后则不用:
```cpp
template <typename T>
BlobPtr<T> BlobPtr<T>::operator++(int) {
  BlobPtr ret = *this;  // save the current value
  ++*this;  // advance one element; prefix ++ checks the increment
  return ret;  // return the saved state
}
```

### (C++11) 模板类型别名
`实例化的 (instantiated)` 模板类是一种具体类型, 可以用 `typedef` 为其定义别名, 而对模板名称则不可以:

```cpp
typedef Blob<string> StrBlob;  // OK
typedef std::map TreeMap;      // error
```

C++11 允许用 `using` 为模板类定义别名:
```cpp
// twin 仍是模板类:
template<typename T> using twin = pair<T, T>;
// authors 的类型是 pair<string, string>
twin<string> authors;
```
这一机制可以用来固定一个或多个模板形参:
```cpp
// 固定第二个类型:
template <typename T> using partNo = pair<T, unsigned>;
// books 的类型是 pair<string, unsigned>
partNo<string> books;
```

## 友元 (少用)

`友元 (friend)` 机制破坏了类的封装, 因此要尽量少用.

如果模板类的友元不是模板, 那么它对该模板的`所有实例化`都是友元.

如果 (模板或非模板) 类的友元本身就是一个模板 (函数或类), 那么友元关系有以下几种可能.

### 一一对应的模板友元
```cpp
// 前置声明:
template <typename> class BlobPtr;
template <typename> class Blob;
template <typename T>
bool operator==(const Blob<T>&, const Blob<T>&);
// 以 Blob 的模板实参作为友元的模板形参:
template <typename T> class Blob {
  // BlobPtr<T> 和 operator==<T> 是 Blob<T> 的友元
  friend class BlobPtr<T>;
  friend bool operator==<T>(const Blob<T>&, const Blob<T>&);
};
```

### 一般与特定的模板友元
一个 (模板或非模板) 类可以指定一个模板类的`所有`或`特定`的实例化作为其友元:
```cpp
// 前置声明:
template <typename T> class Pal;
// 指定 非模板类 的友元:
class C {
  // Pal<C> 是 C 的友元:
  friend class Pal<C>;
  // Pal2<T> 是 C 的友元, 不需要前置声明 Pal2:
  template <typename T> friend class Pal2;
};
// 指定 模板类 的友元:
template <typename T> class C2 {
  // Pal<T> 是 C2<T> 的友元, 需要前置声明 Pal:
  friend class Pal<T>;
  // Pal2<X> 是 C2<T> 的友元, 需要前置声明 Pal2:
  template <typename X> friend class Pal2;
  // Pal3 是 C2<T> 的友元, 不需要前置声明 Pal3:
  friend class Pal3;
};
```

### (C++11) 将模板类型形参设为友元
```cpp
template <typename Type> class Bar {
  friend Type;
};
```

# 模板成员

## 非模板类的模板成员

为`非模板类`定义`模板函数成员`:

```cpp
class DebugDelete {
 public:
  DebugDelete(std::ostream& s = std::cerr) : os(s) { }
  template <typename T> void operator()(T* p) const {
    os << "deleting unique_ptr" << std::endl;
    delete p; 
  }
 private:
  std::ostream& os;
};
```
该类的实例可以用于替代 `delete`:
```cpp
int* ip = new int;
DebugDelete()(ip);  // 临时对象

DebugDelete del;
double* dp = new double;
std::unique_ptr<int, DebugDelete> dp(new int, del);
```

## 模板类的模板成员

为`模板类`声明`模板函数`成员, 二者拥有各自独立的模板形参:

```cpp
template <typename T> class Blob {
  template <typename Iter> Blob(Iter b, Iter e);
};
```
如果在模板类的外部定义模板函数成员, 应当
- 先给出类的模板形参列表
- 再给出成员的模板形参列表
```cpp
template <typename T>
template <typename Iter>
Blob<T>::Blob(Iter b, Iter e)
    : data(std::make_shared<std::vector<T>>(b, e)) {}
```

# 模板形参

## 类型形参

在模板形参列表中, 关键词 `class` 与 `typename` 没有区别:

```cpp
template <typename T, class U>
int calc(const T&, const U&);
```

## 非类型形参
非类型形参的值在`编译期`确定 (人为`指定`或由编译器`推断`), 因此必须为`常量表达式 (constexpr)`:

```cpp
template<unsigned N, unsigned M>
int compare(const char (&p1)[N], const char (&p2)[M]) {
  return strcmp(p1, p2);
}
// 如果以如下方式调用
compare("hi", "mom");
// 该模板将被实例化为
int compare(const char (&p1)[3], const char (&p2)[4]);
```

## 模板形参的作用域
模板形参遵循一般的作用域规则, 但已经被模板形参占用的名字在模板内部`不得`被复用:

```cpp
typedef double A;
template <typename A, typename B> 
void f(A a, B b) {
  A tmp = a;  // tmp 的类型为模板形参 A 而不是 double
  double B;   // 错误: B 已被模板形参占用, 不可复用
}
// 错误: 复用模板形参名
template <typename V, typename V>  // ...
```

## 模板声明
与函数形参名类似, 同一模板的模板形参名在各处声明或定义中不必保持一致.

一个文件所需的所有模板声明, 应当集中出现在该文件头部, 并位于所有用到这些模板名的代码之前.

## 模板形参的类型成员
默认情况下, 编译器认为由 `::` 获得的名字不是一个类型. 因此, 如果要使用模板形参的类型成员, 必须用关键词 `typename` 加以修饰:
```cpp
// T 为一种 容器类型, 并且拥有一个类型成员 value_type
template <typename T>
typename T::value_type top(const T& c) {
  if (!c.empty())
    return c.back();
  else
    return typename T::value_type();
}
```

## (C++11) 默认模板实参
```cpp
template <typename T, typename F = std::less<T>>
int compare(const T& v1, const T& v2, F f = F()) {
  if (f(v1, v2)) return -1;
  if (f(v2, v1)) return 1;
  return 0;
}
```
调用时, 可以 (而非必须) 为其提供一个比较器:

```cpp
bool i = compare(0, 42);
bool j = compare(item1, item2, compareIsbn);
```

如果为所有模板形参都指定了默认模板实参, 并且希望用它们来进行默认实例化, 则必须在模板名后面紧跟 `<>`, 例如:

```cpp
template <class T = int> class Numbers {
 public:
  Numbers(T v = 0): val(v) { }
 private:
  T val; 
};
Numbers<long double> lots_of_precision;
Numbers<> average_precision;  // Numbers<> 相当于 Numbers<int>
```

# 类型参数推导

# 实例化

# 特化

# 可变参数模板

