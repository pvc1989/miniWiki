# 动态内存

# 智能指针

## **`std::unique_ptr`**

`std::unique_ptr` 用于管理`独占所有权`的资源, 具有以下优点:

1. 体积小 --- 默认情况下, 与`裸指针`大小相同.
2. 速度快 --- 大多数操作 (含 `operator*()`) 执行与`裸指针`相同的指令.
3. `move`-only --- 确保独占所有权.



资源析构默认借助于 `operator delete()` 来完成, 也可以在创建时为其指定其他的 deleter. 如果被指定的 deleter 是:

- `函数指针` 或 `含有内部状态的函数对象`, 则 `std::unique_ptr` 的体积比裸指针大.
- `不含有内部状态的函数对象` (例如 无捕获的 lambda 表达式), 则 `std::unique_ptr` 的体积与裸指针相同.



`std::unique_ptr` 的类型参数支持两种形式:

- `std:: unique_ptr<T>` --- 用于单个对象.
- `std::unique_ptr<T[]>` --- 用于对象数组, 几乎只应当用于管理从 C-风格 API 所获得的动态内存.



`std::unique_ptr` 非常适合用作工厂方法的返回类型, 这是因为:

- `std::unique_ptr` 可以很容易地转为 `std::shared_ptr`.
- 将裸指针 (例如返回自 `new`) 赋值给 `std::unique_ptr` 的错误在编译期能够被发现.



## **`make`** 函数

`std::make_shared` 由 C++11 引入, 而 `std::make_unique` 则是由 C++14 引入.



`make` 函数有助于减少代码重复 (例如与 `auto` 配合可以少写一次类型), 并提高`异常安全性`, 例如在如下代码中

```cpp
processWidget(std::unique_ptr<Widget>(new Widget), 
              computePriority());
```

编译器只能保证`参数在被传入函数之前被取值`, 因此实际的运行顺序可能是:

```cpp
new Widget
computePriority()
std::unique_ptr<Widget>()
```

如果第 2 行抛出了异常, 则由 `new` 获得的动态内存来不及被 `std::unique_ptr` 接管, 从而有可能发生泄漏. 改用 `make` 函数则不会发生这种内存泄漏:

```cpp
processWidget(std::make_unique<Widget>(), 
              computePriority());
```

在无法使用 `make` 函数的情况下 (例如指定 deleter 或传入列表初始化参数), 一定要确保由 `new` 获得的动态内存`在一条语句内`被智能指针接管, 并且在该语句内`不做任何其他的事`.



对于 `std::make_shared` 和 `std::allocate_shared`, 用 `make` 函数可以节省空间和运行时间:

```cpp
std::shared_ptr<Widget> spw(new Widget);  // 2 memory allocations
auto spw = std::make_shared<Widget>();    // 1 memory allocation
```



`make` 函数用 `( )` 进行完美转发, 因此无法直接使用对象的列表初始化构造函数. 一种解决办法是先用 `auto` 创建一个 `std::initializer_list` 对象, 再将其传入 `make` 函数:

```cpp
auto initList = { 10, 20 };
auto spv = std::make_shared<std::vector<int>>(initList);
```



对于 `std::shared_ptr`, 不宜使用 `make` 函数的情形还包括: 

- 采用特制内存管理方案的类.
- 系统内存紧张, 对象体积庞大, 且 `std::weak_ptr` 比相应的 `std::shared_ptr` 存活得更久.