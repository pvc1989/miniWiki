#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  int comm_size, comm_rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  printf("Hello from rank %d out of %d processors\n", comm_rank, comm_size);

  MPI_Finalize();
}
