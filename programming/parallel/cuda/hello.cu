#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>  /* CUDA's header */

/* Device code: runs on GPU */
__global__ void Hello() {
  printf("Hello, world from thread[%d][%d][%d] in block[%d][%d][%d]\n",
      threadIdx.x, threadIdx.y, threadIdx.z,
      blockIdx.x, blockIdx.y, blockIdx.z);
}

/* Host code: runs on CPU */
int main(int argc , char* argv[]) {
  /* Get the number of blocks: */
  int n_block = atoi(argv[1]);

  /* Get the number of threads in each block: */
  int n_thread_per_block = atoi(argv[2]);

  /* Invoke the kernel function running on (n_block * n_thread_per_block) threads: */
  Hello<<< n_block, n_thread_per_block >>>();

  /* Wait for GPU to finish: */
  cudaDeviceSynchronize();
  return 0;
}
