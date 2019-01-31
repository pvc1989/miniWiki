# 1. Deducing Types

# 2. **`auto`**

# 3. Moving to Modern C++

# 4. Smart Pointers

## Item 18: Use **`std::unique_ptr`** for exclusive-ownership resource management.

- `std::unique_ptr` is a small, fast, move-only smart pointer for managing resources with exclusive-ownership semantics.

By default, `std::unique_ptr`s are the same size as raw pointers.

For most operations (including dereferencing), they execute exactly the same instructions.

- By default, resource destruction takes place via `delete`, but custom deleters can be specified.

Stateful deleters and function pointers as deleters increase the size of `std::unique_ptr` objects.

Stateless function objects (e.g., from lambda expressions with no captures) incur no size penalty.

- Converting a `std::unique_ptr` to a `std::shared_ptr` is easy.

This is a key part of why `std::unique_ptr` is so well suited as a factory function return type.

Attempting to assign a raw pointer (e.g., from `new`) to a `std::unique_ptr` won’t compile.

- `std::unique_ptr` comes in two forms, one for individual objects (`std:: unique_ptr<T>`) and one for arrays (`std::unique_ptr<T[]>`).

The only situation when a `std::unique_ptr<T[]>` would make sense would be when you’re using a C-like API that returns a raw pointer to a heap array that you assume ownership of.

## Item 19: Use **`std::shared_ptr`** for shared-ownership resource management.

## Item 20: Use **`std::weak_ptr`** for **`std::shared_ptr`**-like pointers that can dangle.

## Item 21: Prefer **`std::make_unique`** and **`std::make_shared`** to direct use of **`new`**.

`std::make_shared` is part of C++11, while `std::make_unique` joined the Standard Library as of C++14.

- `make` functions eliminate source code duplication, and improve exception safety.

```cpp
processWidget(std::unique_ptr<Widget>(new Widget), computePriority());
processWidget(std::make_unique<Widget>(),          computePriority());
```

Make sure that when you use `new` directly, you immediately pass the result to a smart pointer constructor in a statement that *does nothing else*. 

- For `std::make_shared` and `std::allocate_shared`, generate code that’s smaller and faster.

```cpp
std::shared_ptr<Widget> spw(new Widget);  // 2 memory allocations
auto spw = std::make_shared<Widget>();    // 1 memory allocation
```

- Situations where use of `make` functions is inappropriate include the need to specify custom deleters and a desire to pass braced initializers.

Within the `make` functions, the perfect forwarding code uses `( )`, not `{ }`. A workaround is: use `auto` type deduction to create a `std::initializer_list` object, then pass the `auto`-created object through the `make` function.

```cpp
auto initList = { 10, 20 };
auto spv = std::make_shared<std::vector<int>>(initList);
```

- For `std::shared_ptr`s, additional situations where make functions may be ill-advised include
  - classes with custom memory management, and
  - systems with memory concerns, very large objects, and `std::weak_ptr`s that outlive the corresponding `std::shared_ptr`s.

## Item 22: When using the Pimpl Idiom, define special member functions in the implementation file.

# 5. Rvalue References, Move Semantics, and Perfect Forwarding

# 6. Lambda Expressions

# 7. The Concurrency API

# 8. Tweaks

