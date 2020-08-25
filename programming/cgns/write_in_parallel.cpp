/*
  Creates a simple 3d adaptive refined grid.

  Typical compilation and execution (change paths if needed):
    mpicxx ../write_in_parallel.cpp -o write_in_parallel \
      -I/usr/local/include -L/usr/local/lib -lcgns -lhdf5 -std=c++17 &&
    mpiexec -np 2 write_in_parallel 21
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "pcgnslib.h"
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[]) {

  /* initialize MPI */
  int comm_size, comm_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  /* total number of nodes and hex elements */
  assert(argc >= 2);
  int n_nodes_per_side = atoi(argv[1]);
  int n_elems_per_side = n_nodes_per_side - 1;
  assert(n_elems_per_side >= 1);
  int n_nodes_total = n_nodes_per_side * n_nodes_per_side * n_nodes_per_side;
  int n_elems_total = n_elems_per_side * n_elems_per_side * n_elems_per_side;

  /* open the file and create base and zone */
  cgsize_t sizes[3] = {n_nodes_total, n_elems_total, 0};

  /* the default here is to use MPI_COMM_WORLD,
     but this allows assigning of another communicator
  cgp_mpi_comm(MPI_COMM_WORLD);
  */
  auto* file_name = "parallel_grid.cgns";
  int i_file, i_base, i_zone;
  if (cgp_open(file_name, CG_MODE_WRITE, &i_file) ||
      cg_base_write(i_file, "Base", 3, 3, &i_base) ||
      cg_zone_write(i_file, i_base, "Zone", sizes, Unstructured, &i_zone))
    cgp_error_exit();

  /* print info */
  if (comm_rank == 0) {
    printf("writing %d nodes and %d elements to %s\n",
           n_nodes_total, n_elems_total, file_name);
  }

{ /* Grid Coordinates */

  /* create data nodes for coordinates */
  int i_coord_x, i_coord_y, i_coord_z;
  if (cgp_coord_write(i_file, i_base, i_zone, RealSingle,
                      "CoordinateX", &i_coord_x) ||
      cgp_coord_write(i_file, i_base, i_zone, RealSingle,
                      "CoordinateY", &i_coord_y) ||
      cgp_coord_write(i_file, i_base, i_zone, RealSingle,
                      "CoordinateZ", &i_coord_z))
    cgp_error_exit();
 
  /* number of nodes and range this process will write */
  int n_nodes_local = (n_nodes_total + comm_size - 1) / comm_size;
  cgsize_t first = n_nodes_local * comm_rank + 1;
  cgsize_t last = n_nodes_local * (comm_rank + 1);
  if (last > n_nodes_total) last = n_nodes_total;
  
  /* create the coordinate data for this process */
  auto x_head = (float *) malloc(n_nodes_local * sizeof(float));
  auto y_head = (float *) malloc(n_nodes_local * sizeof(float));
  auto z_head = (float *) malloc(n_nodes_local * sizeof(float));
  auto x_curr = x_head;
  auto y_curr = y_head;
  auto z_curr = z_head;
  int i_node_total = 0;
  for (int k = 0; k < n_nodes_per_side; k++) {
    for (int j = 0; j < n_nodes_per_side; j++) {
      for (int i = 0; i < n_nodes_per_side; i++) {
        ++i_node_total;
        if (first <= i_node_total && i_node_total <= last) {
          *x_curr++ = (float) i;
          *y_curr++ = (float) j;
          *z_curr++ = (float) k;
        }
      }
    }
  }

  /* write the coordinate data in parallel */
  if (cgp_coord_write_data(i_file, i_base, i_zone, i_coord_x,
                           &first, &last, x_head) ||
      cgp_coord_write_data(i_file, i_base, i_zone, i_coord_y,
                           &first, &last, y_head) ||
      cgp_coord_write_data(i_file, i_base, i_zone, i_coord_z,
                           &first, &last, z_head))
    cgp_error_exit();
}

{ /* Elements */

  /* create data node for elements */
  int i_elem;
  if (cgp_section_write(i_file, i_base, i_zone, "Hexa", HEXA_8,
                        1, n_elems_total, 0, &i_elem))
    cgp_error_exit();

  /* number of elements and range this process will write */
  int n_elems_local = (n_elems_total + comm_size - 1) / comm_size;
  cgsize_t first = n_elems_local * comm_rank + 1;
  cgsize_t last = n_elems_local * (comm_rank + 1);
  if (last > n_elems_total) last = n_elems_total;

  /* create the hex element data for this process */
  auto elem_head = (cgsize_t *) malloc(8 * n_elems_local * sizeof(cgsize_t));
  auto elem_curr = elem_head;
  int i_elem_total = 0;
  for (int k = 1; k < n_nodes_per_side; k++) {
    for (int j = 1; j < n_nodes_per_side; j++) {
      for (int i = 1; i < n_nodes_per_side; i++) {
        ++i_elem_total;
        if (first <= i_elem_total && i_elem_total <= last) {
          cgsize_t i_node = i + n_nodes_per_side * ((j-1) + n_nodes_per_side * (k-1));
          *elem_curr++ = i_node;
          *elem_curr++ = i_node + 1;
          *elem_curr++ = i_node + 1 + n_nodes_per_side;
          *elem_curr++ = i_node + n_nodes_per_side;
          i_node += n_nodes_per_side * n_nodes_per_side;
          *elem_curr++ = i_node;
          *elem_curr++ = i_node + 1;
          *elem_curr++ = i_node + 1 + n_nodes_per_side;
          *elem_curr++ = i_node + n_nodes_per_side;
        }
      }
    }
  }

  /* write the element connectivity in parallel */
  if (cgp_elements_write_data(i_file, i_base, i_zone, i_elem,
                              first, last, elem_head))
    cgp_error_exit();

  /* Solution and Fields */

  /* create a centered solution */
  int i_sol;
  if (cg_sol_write(i_file, i_base, i_zone, "CellData", CellCenter, &i_sol))
    cgp_error_exit();

  /* create the field data for this process */
  auto cell_index_head = (float *) malloc(n_elems_local * sizeof(float));
  auto cell_index_curr = cell_index_head;
  for (int i_elem_total = 1; i_elem_total <= n_elems_total; i_elem_total++) {
    if (first <= i_elem_total && i_elem_total <= last) {
      *cell_index_curr++ = (float) i_elem_total;
    }
  }

  /* write the solution field data in parallel */
  int i_field;
  if (cgp_field_write(i_file, i_base, i_zone, i_sol, RealSingle,
                      "CellIndex", &i_field) ||
      cgp_field_write_data(i_file, i_base, i_zone, i_sol, i_field,
                           &first, &last, cell_index_head))
    cgp_error_exit();
}

  /* close the file and terminate MPI */
  cgp_close(i_file);  
  MPI_Finalize();
  return 0;
}
