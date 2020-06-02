/* 
  Creates simple 3-D unstructured grid and writes it to a CGNS file.

  Example compilation for this program is (change paths if needed!):

    c++ -std=c++11 -I/usr/local/include -c write.cpp && \
    c++ -o write.exe write.o -L/usr/local/lib -lcgns && \
    ./write.exe && cgnscheck *.cgns && cgnsview *.cgns

  (../../lib is the location where the compiled library libcgns.a is located)
 */

#include <cassert>
#include <cstdio>
#include <cstring>
// cgnslib.h file must be located in directory specified by -I during compile:
#include "cgnslib.h"

#if CGNS_VERSION < 3100
# define cgsize_t int
#endif

int main() {
  /*
    Create A CGNS File
   */
  char file_name[33] = "unstructed.cgns";
  std::printf("A file named \"%s\"\n", file_name);
  std::printf("    is being creating...\n");
  int file_id;
  if (cg_open(file_name, CG_MODE_WRITE, &file_id))
    cg_error_exit();
  std::printf("    has been reated with id %d.\n", file_id);
  /*
    Create A CGNSBase_t Node
   */
  char base_name[33] = "SimpleBase";
  std::printf("A `CGNSBase_t` named \"%s\"\n", base_name);
  std::printf("    is being creating...\n");
  int cell_dim{3}, phys_dim{3};
  int base_id;
  cg_base_write(file_id, base_name, cell_dim, phys_dim, &base_id);
  std::printf("    has been created with id %d.\n", base_id);
  /*
    Create A Zone_t Node
   */
  char zone_name[33];
  strcpy(zone_name, "HexaZone");
  std::printf("A `Zone_t` named \"%s\"\n", zone_name);
  std::printf("    is being creating...\n");
  cgsize_t grid_size[3][1];
  constexpr int ni{21}, nj{17}, nk{9};
  grid_size[0][0] = ni * nj * nk;  // vertex size
  constexpr int n_hexa_elems = (ni-1) * (nj-1) * (nk-1);
  grid_size[1][0] = n_hexa_elems;  // cell size
  grid_size[2][0] = 0;  // boundary vertex size (zero if elements not sorted)
  int zone_id;
  cg_zone_write(file_id, base_id, zone_name, grid_size[0],
      CGNS_ENUMV(Unstructured), &zone_id);
  std::printf("    has been created with id %d.\n", zone_id);
  /*
    Create A GridCoordinates_t Node
   */
  std::printf("A `GridCoordinates_t` is being creating...\n");
  double x[nk][nj][ni], y[nk][nj][ni], z[nk * nj * ni];
  int i_node = 0;
  for (int k = 0; k < nk; k++) {
    for (int j = 0; j < nj; j++) {
      for (int i = 0; i < ni; i++) {
        x[k][j][i] = i - 1.;
        y[k][j][i] = j - 1.;
        assert((k * nj + j) * ni + i == i_node);
        z[i_node] = k - 1.;
        ++i_node;
      }
    }
  }
  assert(i_node == grid_size[0][0]);
  int coord_id;
  // user must use SIDS-standard names (e.g. "CoordinateX") here:
  char coord_name[33] = "CoordinateX";
  cg_coord_write(file_id, base_id, zone_id,
      CGNS_ENUMV(RealDouble), coord_name, x, &coord_id);
  std::printf("    A `DataArray_t` named \"%s\" has been created with id %d.\n",
      coord_name, coord_id);
  std::strcpy(coord_name, "CoordinateY");
  cg_coord_write(file_id, base_id, zone_id,
      CGNS_ENUMV(RealDouble), coord_name, y, &coord_id);
  std::printf("    A `DataArray_t` named \"%s\" has been created with id %d.\n",
      coord_name, coord_id);
  std::strcpy(coord_name, "CoordinateZ");
  cg_coord_write(file_id, base_id, zone_id,
      CGNS_ENUMV(RealDouble), coord_name, z, &coord_id);
  std::printf("    A `DataArray_t` named \"%s\" has been created with id %d.\n",
      coord_name, coord_id);
  /*
    Create A Elements_t Node for Interior
   */
  // set interior HEXA_8 elements (mandatory):
  char section_name[33] = "InteriorHexa";
  std::printf("An `Elements_t` named \"%s\"", section_name);
  std::printf(" is being creating...\n");
  cgsize_t hexa_elems[n_hexa_elems][8];
  int section_id;
  int i_first_node;
  int i_elem = 0;
  for (int k = 1; k < nk; k++) {
    for (int j = 1; j < nj; j++) {
      for (int i = 1; i < ni; i++) {
        i_first_node = i + (j-1) * ni + (k-1) * ni * nj;
        hexa_elems[i_elem][0] = i_first_node;
        hexa_elems[i_elem][1] = i_first_node + 1;
        hexa_elems[i_elem][2] = i_first_node + 1 + ni;
        hexa_elems[i_elem][3] = i_first_node + ni;
        hexa_elems[i_elem][4] = i_first_node + ni * nj;
        hexa_elems[i_elem][5] = i_first_node + ni * nj + 1;
        hexa_elems[i_elem][6] = i_first_node + ni * nj + 1 + ni;
        hexa_elems[i_elem][7] = i_first_node + ni * nj + ni;
        ++i_elem;
      }
    }
  }
  assert(i_elem == grid_size[1][0]);
  cgsize_t i_elem_first{1}, i_elem_last{i_elem};
  int n_boundary_elements = 0;
  cg_section_write(file_id, base_id, zone_id,
      section_name, CGNS_ENUMV(HEXA_8), i_elem_first, i_elem_last,
      n_boundary_elements, hexa_elems[0], &section_id);
  std::printf("    has been created with id %d.\n", section_id);
  /*
    Create Elements_t Nodes for Boundaries
   */
  // set boundary QUAD_4 elements (optional):
  constexpr int n_quad_elems = (ni * nj + nj * nk + nk * ni) * 2;
  cgsize_t quad_elems[n_quad_elems][4];
  // Inflow:
  std::strcpy(section_name, "InflowQuad");
  std::printf("An `Elements_t` named \"%s\"", section_name);
  std::printf(" is being creating...\n");
  i_elem = 0;
  int i = 1;
  for (int k = 1; k < nk; k++) {
    for (int j = 1; j < nj; j++) {
      i_first_node = ((k-1) * nj + (j-1)) * ni + i;
      quad_elems[i_elem][0] = i_first_node;
      quad_elems[i_elem][1] = quad_elems[i_elem][0] + ni * nj;
      quad_elems[i_elem][2] = quad_elems[i_elem][1] + ni;
      quad_elems[i_elem][3] = quad_elems[i_elem][0] + ni;
      ++i_elem;
    }
  }
  i_elem_first = i_elem_last;
  i_elem_last = i_elem_first + i_elem;
  ++i_elem_first;
  cg_section_write(file_id, base_id, zone_id,
      section_name, CGNS_ENUMV(QUAD_4), i_elem_first, i_elem_last,
      n_boundary_elements, quad_elems[0], &section_id);
  std::printf("    has been created with id %d.\n", section_id);
  // Outflow:
  std::strcpy(section_name, "OutflowQuad");  
  std::printf("An `Elements_t` named \"%s\"", section_name);
  std::printf(" is being creating...\n");
  i_elem = 0;
  i_elem_first = i_elem_last + 1;
  i = ni - 1;
  for (int k = 1; k < nk; k++) {
    for (int j = 1; j < nj; j++) {
      i_first_node = ((k-1) * nj + (j-1)) * ni + i;
      quad_elems[i_elem][0] = i_first_node + 1;
      quad_elems[i_elem][1] = quad_elems[i_elem][0] + ni;
      quad_elems[i_elem][2] = quad_elems[i_elem][1] + ni * nj;
      quad_elems[i_elem][3] = quad_elems[i_elem][2] - ni;
      ++i_elem;
    }
  }
  i_elem_last = i_elem_first + i_elem - 1;
  // write QUAD element connectivity for outflow face (user can give any name):
  cg_section_write(file_id, base_id, zone_id,
      section_name, CGNS_ENUMV(QUAD_4), i_elem_first, i_elem_last,
      n_boundary_elements, quad_elems[0], &section_id);
  std::printf("    has been created with id %d.\n", section_id);
  /*
    Create A FlowSolution_t Node
   */
  {
    int sol_id;
    char sol_name[33] = "NodeData";
    cg_sol_write(file_id, base_id, zone_id,
        sol_name, CGNS_ENUMV(Vertex), &sol_id);
    int field_id;
    char field_name[33] = "Pressure";
    double data[nk][nj][ni];
    for (int k = 0; k < nk; k++) {
      for (int j = 0; j < nj; j++) {
        for (int i = 0; i < ni; i++) {
          data[k][j][i] = i * 2.5;
        }
      }
    }
    cg_field_write(file_id, base_id, zone_id, sol_id,
        CGNS_ENUMV(RealDouble), field_name, data[0][0], &field_id);
  }
  // density at cell centers
  {
    int sol_id;
    char sol_name[33] = "CellData";
    cg_sol_write(file_id, base_id, zone_id,
        sol_name, CGNS_ENUMV(CellCenter), &sol_id);
    int field_id;
    char field_name[33] = "Density";
    double data[n_hexa_elems];
    i_elem = 0;
    for (int k = 0; k < nk-1; k++) {
      for (int j = 0; j < nj-1; j++) {
        for (int i = 0; i < ni-1; i++) {
          data[i_elem++] = j * 2.5;
        }
      }
    }
    assert(n_hexa_elems == i_elem);
    cg_field_write(file_id, base_id, zone_id, sol_id,
        CGNS_ENUMV(RealDouble), field_name, data, &field_id);
    // partial write:
    i_elem = nk/2 * (nj-1) * (ni-1);
    cgsize_t range_min = i_elem + 1;  // SIDS's indexing is 1-based.
    cgsize_t range_max = range_min + (nj-1) * (ni-1);
    std::strcpy(field_name, "DensityPartiallyWritten");
    cg_field_partial_write(file_id, base_id, zone_id, sol_id,
        CGNS_ENUMV(RealDouble), field_name,
        &range_min, &range_max, &data[i_elem], &field_id);
  }
  /*
    Close the CGNS File
   */
  std::printf("Closing the CGNS file...\n");
  cg_close(file_id);
  std::printf("Successfully wrote unstructured grid to \"%s\".\n", file_name);
  return 0;
}
