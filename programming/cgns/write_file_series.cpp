/* 
  Creates a simple 3d adaptive refined grid.

  Example compilation for this program is (change paths if needed!):
    c++ -std=c++11 -o write_file_series.exe ../write_file_series.cpp \
      -I/usr/local/include -L/usr/local/lib -lcgns && \
    ./write_file_series.exe 3 && \
    cgnscheck step#0.cgns && cgnscheck step#1.cgns && cgnscheck step#2.cgns
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
// cgnslib.h file must be located in directory specified by -I during compile:
#include "cgnslib.h"

#if CGNS_VERSION < 3100
# define cgsize_t int
#endif

int write_step(int step) {
  constexpr int kNameLength = 32;
  /*
    Create A CGNS File
   */
  // set file name:
  char file_name[kNameLength + 1];
  sprintf(file_name, "step#%d.cgns", step);
  std::printf("A file named \"%s\"\n", file_name);
  std::printf("    is being creating... ");
  // get file id:
  int file_id;
  if (cg_open(file_name, CG_MODE_WRITE, &file_id))  // check the returned ierror
    cg_error_exit();
  std::printf("has been created with id %d.\n", file_id);
  /*
    Create A CGNSBase_t
   */
  // set base name:
  char base_name[kNameLength + 1];
  sprintf(base_name, "SimpleBase");
  std::printf("A CGNSBase_t named \"%s\"\n", base_name);
  std::printf("    is being creating... ");
  // set base dims:
  int cell_dim{2}, phys_dim{3};
  // get base id:
  int base_id;
  if (cg_base_write(file_id, base_name, cell_dim, phys_dim, &base_id))
    cg_error_exit();
  std::printf("has been created with id %d.\n", base_id);
  // set simulation type:
  if (cg_simulation_type_write(file_id, base_id, CGNS_ENUMV(TimeAccurate)))
    cg_error_exit();
  /*
    Create current step
   */
  // set zone name:
  auto zone_name = "SimpleZone";
  std::printf("A Zone_t named \"%s\"\n", zone_name);
  std::printf("    is being creating... ");
  // set zone size:
  int n_cells_x{1 << step}, n_cells_y{2 << step};
  int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  int n_cells = n_cells_x * n_cells_y;
  int n_nodes = n_nodes_x * n_nodes_y;
  cgsize_t grid_size[3][1] = {n_nodes, n_cells, 0};
  // get zone id:
  int zone_id;
  if (cg_zone_write(file_id, base_id, zone_name, grid_size[0],
      CGNS_ENUMV(Unstructured), &zone_id))
    cg_error_exit();
  std::printf("has been created with id %d.\n", zone_id);
  // set nodes (coordinates):
  double dx = 1.0 / n_cells_x;
  double dy = 2.0 / n_cells_y;
  auto coord_x = std::vector<double>(n_nodes);
  auto coord_y = std::vector<double>(n_nodes);
  auto coord_z = std::vector<double>(n_nodes, 1.0);
  int i = 0;
  double x = -dx + 1.1 * step;
  for (int ix = 0; ix != n_nodes_x; ++ix) {
    x += dx;
    double y = -dy;
    for (int iy = 0; iy != n_nodes_y; ++iy) {
      y += dy;
      coord_x[i] = x;
      coord_y[i] = y;
      ++i;
    }
    assert(y == 2.0);
  }
  assert(i == n_nodes);
  int coord_id;
  if (cg_coord_write(file_id, base_id, zone_id,
      CGNS_ENUMV(RealDouble), "CoordinateX", coord_x.data(), &coord_id))
    cg_error_exit();
  if (cg_coord_write(file_id, base_id, zone_id,
      CGNS_ENUMV(RealDouble), "CoordinateY", coord_y.data(), &coord_id))
    cg_error_exit();
  if (cg_coord_write(file_id, base_id, zone_id,
      CGNS_ENUMV(RealDouble), "CoordinateZ", coord_z.data(), &coord_id))
    cg_error_exit();
  // set cells (connectivities):
  char section_name[kNameLength+1] = "Interior";
  std::printf("An Elements_t named \"%s\"", section_name);
  std::printf(" is being creating... ");
  cgsize_t quad_elems[n_cells][4];
  int section_id;
  int i_elem = 0;
  for (int ix = 0; ix < n_cells_x; ix++) {
    for (int iy = 0; iy < n_cells_y; iy++) {
      quad_elems[i_elem][0] = n_nodes_y * ix + iy + 1;
      quad_elems[i_elem][1] = quad_elems[i_elem][0] + 1;
      quad_elems[i_elem][2] = quad_elems[i_elem][1] + n_nodes_y;
      quad_elems[i_elem][3] = quad_elems[i_elem][2] - 1;
      ++i_elem;
    }
  }
  assert(i_elem == grid_size[1][0]);
  cgsize_t i_elem_first{1}, i_elem_last{i_elem};
  cg_section_write(file_id, base_id, zone_id,
      section_name, CGNS_ENUMV(QUAD_4), i_elem_first, i_elem_last,
      0/* n_boundary_elements */, quad_elems[0], &section_id);
  std::printf("    has been created with id %d.\n", section_id);
  // set node data:
  // set cell data:
  /*
    Close the CGNS File
   */
  std::printf("\"%s\" is being closing... ", file_name);
  if (cg_close(file_id))
    cg_error_exit();
  std::printf("has been closed.\n\n");
  return 0;
}

int main(int argc, char* argv[]) {
  assert(argc == 2);
  int n_steps = atoi(argv[1]);
  assert(n_steps < 8);
  for (int i = 0; i < n_steps; i++)
    write_step(i);
  return 0;
}
