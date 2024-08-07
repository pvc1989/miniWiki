---
title: CUDA
---

- [Programming Guides](https://docs.nvidia.com/cuda/#programming-guides)
  - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
  - [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)
- [CUDA API References](https://docs.nvidia.com/cuda/#cuda-api-references)
  - [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api)
  - [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api)
  - [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api)

# Thread Hierarchy

|            物理单元             | 逻辑单元 |
| :-----------------------------: | :------: |
| *G*raphics *P*rocessing *U*nits |  `grid`  |
|  *S*treaming *M*ultiprocessor   | `block`  |
|     *S*treaming *P*rocessor     | `thread` |

每个 `thread` 启动时，有以下 `dim3` 型变量可用：

- `threadIdx` 用于在当前 `block` 内索引 `thread`
- `blockIdx` 用于在当前 `grid` 内索引 `block`
- `blockDim` 用于表示当前 `block` 的维数，即其所含 `thread`s 的数量
- `gridDim` 用于表示当前 `grid` 的维数，即其所含 `block`s 的数量

类型 `dim3` 的定义大致为

```c++
struct dim3 {
  int x, y, z;

  dim3(int xx = 1, int yy = 1, int zz = 1)
      : x(xx), y(yy), z(zz) {}
};
```

## *Hello, world* in CUDA

[`hello.cu`](./cuda/hello.cu) 关键行：

```c
/* ... */
#include <cuda.h>  /* CUDA's header */

/* Device code: runs on GPU */
__global__ void Hello() {
  printf("Hello, world from thread[%d][%d][%d] in block[%d][%d][%d]\n",
      threadIdx.x, threadIdx.y, threadIdx.z,
      blockIdx.x, blockIdx.y, blockIdx.z);
}

/* Host code: runs on CPU */
int main(int argc , char* argv[]) {
  /* ... */

  /* Invoke the kernel function running on (n_block * n_thread_per_block) threads: */
  Hello<<< n_block, n_thread_per_block >>>();

  /* Wait for GPU to finish: */
  cudaDeviceSynchronize();

  /* ... */
}
```

编译、运行：

```shell
nvcc -o hello.exe hello.cu
./hello.exe 4 2
```

可能的结果：

```
$ ./hello.exe 4 2
Hello, world from thread[0][0][0] in block[0][0][0]
Hello, world from thread[1][0][0] in block[0][0][0]
Hello, world from thread[0][0][0] in block[3][0][0]
Hello, world from thread[1][0][0] in block[3][0][0]
Hello, world from thread[0][0][0] in block[2][0][0]
Hello, world from thread[1][0][0] in block[2][0][0]
Hello, world from thread[0][0][0] in block[1][0][0]
Hello, world from thread[1][0][0] in block[1][0][0]
```

## Block ID

```c
// linear index of current block in its grid
if (grid_dim == 1) {
  return blockIdx.x;
} else if (grid_dim == 2) {
  return blockIdx.x + blockIdx.y * gridDim.x;
} else if (grid_dim == 3) {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
}
```

## Thread ID

```c
// linear index of current thread in its block
if (block_dim == 1) {
  return threadIdx.x;
} else if (block_dim == 2) {
  return threadIdx.x + threadIdx.y * blockDim.x;
} else if (block_dim == 3) {
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
}
```

# Memory Hierarchy

## Device Memory

```c
__host__ __device__ cudaError_t cudaMalloc(void **devPtr, size_t size);

__host__ __device__ cudaError_t cudaFree(void *ptr)

/* 自带 sync */
__host__ ​cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
    cudaMemcpyKind kind/* cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost */);
```

## Unified Memory

```c
__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size
    unsigned flags = cudaMemAttachGlobal);
```

- 获得的内存可以被 host 或 device 使用，但需要被 `cudaFree()` 释放。
- 只少需要 `compuate capability >= 3.0`，但若 `compuate capability < 6.0`，则不能同时被 host 与 device 访问。
- 使用 unified memory 的 kernel 可能比只使用 device memory 的 kernel 慢。

利用全局 `__managed__` 变量获得 kernel 计算结果：

```c
__managed__ ini sum;

__global__ void Add(int x, int y) {
  sum = x + y;  // 在 device 中更新
}

int main() {
  sum = -5;  // 在 host 中初始化
  Add<<< 1, 1 >>>(2, 5);
  cudaDeviceSynchronize();
  printf("sum == %d\n", sum);  // 在 host 中输出结果
  return 0; 
}
```
