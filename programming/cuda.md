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

# [Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

| Compute Capability | Architecture Name | Release Year |
| :----------------: | :---------------: | :----------: |
|        9.0         |      Hopper       |     2022     |
|        8.0         |      Ampere       |     2020     |
|        7.5         |      Turing       |     2018     |
|        7.0         |       Volta       |     2017     |
|        6.x         |      Pascal       |     2016     |
|        5.x         |      Maxwell      |     2014     |
|        3.0         |      Kepler       |     2012     |
|        2.0         |       Fermi       |     2010     |
|        1.0         |       Tesla       |     2006     |

# Thread Hierarchy

|            物理单元             | 逻辑单元 |
| :-----------------------------: | :------: |
| *G*raphics *P*rocessing *U*nits | **grid** |
|  *S*treaming *M*ultiprocessor   | **block** |
|     *S*treaming *P*rocessor     | **thread** |

同一 *grid* 内的各 *block*s 同构且相互独立，以允许不同的运行顺序：

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/automatic-scalability.png)

每个 *thread* 启动时，有以下 `dim3` 型变量可用：

- `threadIdx` 用于在当前 *block* 内索引 *thread*
- `blockIdx` 用于在当前 *grid* 内索引 *block*
- `blockDim` 用于表示当前 *block* 的维数，即其所含 *thread*s 的数量
- `gridDim` 用于表示当前 *grid* 的维数，即其所含 *block*s 的数量

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

## Thread and Block Indexing

相当于对 C 数组 `T a[dim.z][dim.y][dim.x]` 按 `a[idx.z][idx.y][idx.x]` 索引的地址偏移。

```c
size_t _ID(dim3 idx, dim3 dim) {
  return idx.x + (idx.y + idx.z * dim.y) * dim.x;
}

size_t threadID() {
  return _ID(threadIdx, blockDim);
}

size_t blockID() {
  return _ID(blockIdx, gridDim);
}
```

## [Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)

定义：不可分的*读-改-写*操作。

后缀决定该操作原子性的 [**scope**](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes)：
- 不带后缀的版本，*scope* 为 `cuda::thread_scope_device`
- 后缀为 `_block` 的版本，*scope* 为 `cuda::thread_scope_block`，需要 `compute capability >= 6.0`
- 后缀为 `_system` 的版本，*scope* 为 `cuda::thread_scope_system`，需要 `compute capability >= 7.2`

```c
// compute capability >= 2.x
__device__ float atomicAdd(float* address, float val);

// compute capability >= 6.x
__device__ double atomicAdd(double* address, double val);
```

```c
// Compare-And-Swap
T atomicCAS(T* addr, T comp, T val) {
  // Effectively do the following in an atomic way.
  T old = *addr;
  *addr = (old == comp ? val : old);
  return old;
}
```

|     Name     |                     Effect                      |
| :----------: | :---------------------------------------------: |
| `atomicAdd`  |                 `*addr += val`                  |
| `atomicSub`  |                 `*addr -= val`                  |
| `atomicExch` |                  `*addr = val`                  |
| `atomicMin`  |            `*addr = min(*addr, val)`            |
| `atomicMax`  |            `*addr = max(*addr, val)`            |
| `atomicInc`  |       `*addr < val ? ++*addr : *addr = 0`       |
| `atomicDec`  | `*addr && *addr <= val ? --*addr : *addr = val` |
| `atomicAdd`  |                 `*addr &= val`                  |
|  `atomicOr`  |                 `*addr |= val`                  |
| `atomicXor`  |                 `*addr ^= val`                  |

## Fast Barrier

CUDA 提供一种高效同步同一 *block* 内所有 *thread*s 的机制：

```c
__device__ void __syncthreads(void);
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

## Warp Shuffle

> `compuate capability >= 3.0`

具有连续 `threadID` 且属于同一 *block* 的一组 *thread*s，以 SIMD 模式运行（不同 *warp*s 可以独立运行）。

```c
int warpSize  /* currently 32 */

/* 当前 thread 在其所属 warp 内的编号： */
int lane = threadID % warpSize;
```

同一 *warp* 内的*thread*s 可以读取其他 *thread*s 的寄存器：

```c
__device__ float __shfl_down_sync(
    unsigned mask/* 参与 shuffle 的 threads, e.g. 0xFFFFFFFF */,
    float val/* 当前 thread（即 lane = lane_caller）传入的值 */,
    unsigned diff/* 返回 lane = lane_caller + diff 传入的 val */,
    int witdh = warpSize)
```

⚠️ 若 `lane_caller + diff`
- 未调用该函数，则返回值未定义⚠️
- `> max_lane_in_this_warp`，则返回值未定义⚠️
- `>= warpSize`，则返回当前 *thread* 传入的 `val`

```c
/* 所有 threads 都参与求和，只有 land = 0 的那个 thread 的返回值正确 */
__device__ float WarpSum(float val) {
  for (int diff = warpSize / 2; diff > 0; diff /= 2) {
    var += __shfl_down_sync(0xFFFFFFFF, var, diff);
  }
  return var;
}
```

## Shared Memory

由同一 SM 内的 SPs 共享的高速缓存，逻辑上分为 32 **bank**s（`a[i]` 与 `a[i + 32]` 属于同一 *bank*）：
- 若同一 *warp* 内的所有 *thread*s 访问的 *bank*s 都不相同，则可并行访问。
- 若多个 *thread*s 访问同一 *bank* 内的相同地址，则可并行访问。
- 若多个 *thread*s 访问同一 *bank* 内的不同地址，则必须串行化。

```c
/* 模仿 WarpSum，所有 threads 的返回值都正确 */
__device__ float SharedSum(float vals[]) {
  int my_lane = threadID() % warpSize;
  for (int diff = warpSize / 2; diff > 0; diff /= 2) {
    int source = (lane + diff) % warpSize;
    vars[my_lane] += vars[source];  /* 以 SIMD 方式运行，没有竞争 */
  }
  return vars[my_lane];
}
```

其中 `vals` 应显式定义在 shared memory 中：

```c
// 显式指定数组大小
__shared__ float vals[32];

// 运行期由 SharedSum<<< n_block, n_thread_per_block, n_byte >>> 指定数组大小
extern __shared__ float vals[];
```

# Streams and (Grid Level) Concurrency

CUDA API 分两类：
- 【同步的】指调用后阻塞 host，直到该操作运行结束。
- 【异步的】指调用后返回 host，实用其结果前需显式同步。

## `cudaStream_t`

【Stream】指由 host 发起、在 device 上运行的*异步*操作（含 host--device 数据迁移、kernel 启动等）序列。
- 同一 stream 内部，操作按 host 指定的顺序执行。
- 不同 streams 之间的操作没有严格顺序（除非人为引入依赖）。

Streams 分两类：
- 隐式声明的 NULL stream（默认）
- 显式声明的 non-NULL stream（粗粒度并发所必需）

典型用例：

```c
// 在 host 上分配 pinned (non-pageable) memory：
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);

// 构造 cudaStream_t：
cudaStream_t stream[kStreams];
for (int i = 0; i < kStreams; i++) {
  cudaStreamCreate(streams + i);
}

for (int i = 0; i < kStreams; i++) {
  // 异步地 host-to-device 迁移数据：
  cudaMemcpyAsync(device_mem, pinned_host_mem, n_byte,
      cudaMemcpyHostToDevice, streams[i]);

  // 启动 kernel：
  MyKernel<<<grid, block, sharedMemSize, stream>>>(args);

  // 异步地 device-to-host 迁移数据：
  cudaMemcpyAsync(pinned_host_mem, device_mem, n_byte,
      cudaMemcpyDeviceToHost, streams[i]);
}

for (int i = 0; i < kStreams; i++) {
  // 显式同步：
  cudaStreamSynchronize(streams[i]);
      // 阻塞，直到该 stream 中的操作全部完成
  cudaStreamQuery(streams[i]);
      // 非阻塞，返回 cudaSuccess 或 cudaErrorNotReady
}

// 析构 cudaStream_t：
for (int i = 0; i < kStreams; i++) {
  cudaStreamDestroy(streams[i]);
}
```

⚠️ 受 PCIe bus 限制，同一 device 上至多可同时执行 1 个 host-to-device、1 个 device-to-host 数据传输操作。
