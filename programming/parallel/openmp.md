---
title: OpenMP
---

# [Hello](./openmp/hello.cpp)

```cpp
// ...
#include <omp.h>

// the thread function
void hello() {
  int rank = omp_get_thread_num();
  int size = omp_get_num_threads();
  // ...
}

int main(int argc, char *argv[]) {
  int n_thread = std::atoi(argv[1]);

# pragma omp parallel num_threads(n_thread)
  hello();

  return 0;
}
```
