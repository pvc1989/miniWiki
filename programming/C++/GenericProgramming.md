# 模板函数
`T` 的实际类型将根据 `compare` 的`静态`调用方式在`编译期`决定:

```cpp
template <typename T>  // 模板参数列表
int compare(const T& v1, const T& v2) {
  if (v1 < v2) return -1;
  if (v2 < v1) return 1;
  return 0;
}
```

`inline`, `constexpr` 等修饰符应当位于`模板参数列表`与`返回值类型`之间:
```cpp
template <typename T>
inline T min(const T&, const T&);
```

编写模板函数应当
- 尽量减少对参数类型的要求.
- 用`指向常量的引用`作为函数参数类型,  这样可以使得代码也适用于不可拷贝的类型.
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
这一机制可以用来固定一个或多个模板参数:
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

### (C++11) 将模板类型参数设为友元
```cpp
template <typename Type> class Bar {
  friend Type;
};
```


# 模板参数

# 类型参数推导

# 实例化

# 特化

# 可变参数模板

