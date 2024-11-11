#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
  int comm_size, comm_rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  std::cout << "Hello from rank " << comm_rank << " out of " 
            << comm_size << "processors" << std::endl;

  MPI_Finalize();
}
