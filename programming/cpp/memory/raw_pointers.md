# 原始指针
## `new`

### 分配内存 + 构造对象
置于「类型名」之前的 `new` 运算符用于创建「单个动态对象」。
如果分配成功，则返回一个「指向动态对象的指针」，否则抛出异常：
```cpp
int* p = new int;
```
`new` 语句依次完成三个任务:
1. 动态分配所需内存；
2. 默认初始化对象；
3. 返回指向该对象的「原始 (raw) 指针」。

### 值初始化
默认情况下，动态分配对象时采用的是「默认 (default) 初始化」。
若要进行「值 (value) 初始化」，需要在「类型名」后面紧跟 `()` 或 `{}`，例如
```cpp
std::string* ps1 = new std::string;    // 默认初始化 为 空字符串
std::string* ps2 = new std::string();  // 值初始化 为 空字符串
int* pi1 = new int;    // 默认初始化 为 不确定值
int* pi2 = new int();  // 值初始化 为 0
```

### 常值对象
动态分配的常值对象必须由「指向常量的指针」接管，并且在创建时被初始化：
```cpp
const int* pci = new const int(1024);
const std::string* pcs = new const std::string;
```
自 C++11 起，推荐用 `auto` 作为对象类型，编译器会推断出变量的类型：
```cpp
auto pi = new int();
auto ps = new std::string();
auto pci = new const int(1024);
auto pcs = new const std::string("hello");
```

### 内存耗尽
内存空间在运行期有可能被耗尽，此时「分配内存」的任务无法完成。
- 在默认情况下，分配失败会抛出 `std::bad_alloc`。
- 如果在 `new` 与「类型名」之间插入 `(std::nothrow)`，则分配失败时不会抛出异常，而是以 `nullptr` 作为返回值。⚠️ 使用这种形式一定要记得检查返回值是否为 `nullptr`。
- `std::bad_alloc` 及 `std::nothrow` 都定义在 `<new>` 中。
```cpp
#include <new>
int* p1 = new int;                 // 如果分配失败, 将抛出 std::bad_alloc
int* p2 = new (std::nothrow) int;  // 如果分配失败, 将返回 nullptr
```
## `delete`

### 析构对象 + 释放内存
传递给 `delete` 的指针必须是「指向动态对象的指针」或 `nullptr`：
```cpp
delete p;     // 析构并释放 (单个) 动态对象
delete[] pa;  // 析构并释放 (整个) 动态数组
```

⚠️ 编译器无法判断一个指针所指的「对象是否是动态的」，也无法判断一个指针所指的「内存是否已经被释放」。

### 内存泄漏
```cpp
Foo* factory(T arg) { return new Foo(arg); }
void use_factory(T arg) {
  Foo *p = factory(arg)  // factory 返回一个指向动态内存的指针
  // 使用 p
  delete p;  // 调用者负责将其释放
}
```
如果 `use_factory` 在返回前没有释放 `p` 所指向的动态内存，则 `use_factory` 的调用者将不再有机会将其释放，可用的动态内存空间将会变小。
这种现象被称为「内存泄漏 (memory leak)」。

### 空悬指针
在执行完 `delete p;` 之后, `p` 将成为一个「空悬 (dangling) 指针」，对其进行
- 解引用，并进行
  - 读：返回无意义的值。
  - 写：有可能破坏数据。
- 二次 `delete`：会破坏内存空间。

为避免这些陷阱，应当
- 将 `delete p;` 尽可能放在 `p` 的作用域末端，或者
- 在 `delete p;` 后面紧跟 `p = nullptr;`。

即便如此，由于同一个动态对象有可能被多个指针所指向，还是会有危险：
```cpp
auto q = p;   // p 和 q 指向同一块动态内存
delete p;     // 释放
p = nullptr;  // p 不再指向该地址
              // q 仍然指向该地址, 对其进行 解引用 或 二次释放 都有可能造成破坏
```
