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
