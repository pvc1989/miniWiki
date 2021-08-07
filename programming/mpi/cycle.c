#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
	int comm_rank, comm_size, value = 0;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_rank != 0) {
		MPI_Recv(&value, 1, MPI_INT, comm_rank - 1, 0, MPI_COMM_WORLD,
						 MPI_STATUS_IGNORE);
		printf("Process %d received %d from process %d\n", comm_rank, value,
					 comm_rank - 1);
    value++;
	}
  /* For rank != 0, MPI_Send() is called after MPI_Recv() returns. */
	MPI_Send(&value, 1, MPI_INT, (comm_rank + 1) % comm_size, 0,
					 MPI_COMM_WORLD);
  /* For rank == 0, MPI_Recv() is called after MPI_Send() returns. */
	if (comm_rank == 0) {
		MPI_Recv(&value, 1, MPI_INT, comm_size - 1, 0, MPI_COMM_WORLD,
						 MPI_STATUS_IGNORE);
		printf("Process %d received %d from process %d\n", comm_rank, value,
					 comm_size - 1);
	}

	MPI_Finalize();
}
