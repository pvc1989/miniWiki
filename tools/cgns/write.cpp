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
#include <string>
// cgnslib.h file must be located in directory specified by -I during compile:
#include "cgnslib.h"

#if CGNS_VERSION < 3100
# define cgsize_t int
#endif

int main() {
  constexpr int kNameLength = 32;
  /*
    Create A CGNS File
   */
  char file_name[kNameLength+1] = "unstructed.cgns";
  std::printf("A file named \"%s\"\n", file_name);
  std::printf("    is being creating...\n");
  int file_id;
  if (cg_open(file_name, CG_MODE_WRITE, &file_id))
    cg_error_exit();
  std::printf("    has been reated with id %d.\n", file_id);
  /*
    Create A CGNSBase_t Node
   */
  char base_name[kNameLength+1] = "SimpleBase";
  std::printf("A `CGNSBase_t` named \"%s\"\n", base_name);
  std::printf("    is being creating...\n");
  int cell_dim{3}, phys_dim{3};
  int base_id;
  cg_base_write(file_id, base_name, cell_dim, phys_dim, &base_id);
  std::printf("    has been created with id %d.\n", base_id);
  /*
    Create A Zone_t Node
   */
  char zone_name[kNameLength+1];
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
    Write Vertex Coordinates
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
  char coord_name[kNameLength+1] = "CoordinateX";
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
    Write Interior Elements
   */
  // set interior HEXA_8 elements (mandatory):
  char section_name[kNameLength+1] = "InteriorHexa";
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
    Write Boundary Elements
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
    Write Time-Independent Flow Solutions
   */
  {
    int sol_id;
    char sol_name[kNameLength+1] = "NodeData";
    cg_sol_write(file_id, base_id, zone_id,
        sol_name, CGNS_ENUMV(Vertex), &sol_id);
    int field_id;
    char field_name[kNameLength+1] = "Pressure";
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
    char sol_name[kNameLength+1] = "CellData";
    cg_sol_write(file_id, base_id, zone_id,
        sol_name, CGNS_ENUMV(CellCenter), &sol_id);
    int field_id;
    char field_name[kNameLength+1] = "Density";
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
  }
  /*
    Write Time-Dependent Flow Solutions
   */
  {
    constexpr int kSteps = nk - 1;
    // set time steps:
    double time[kSteps];
    for (int k = 0; k < kSteps; ++k) {
      time[k] = k * 0.1;
    }
    // set data to be write:
    double data[n_hexa_elems];
    i_elem = 0;
    for (int k = 1; k < nk; k++) {
      for (int j = 1; j < nj; j++) {
        for (int i = 1; i < ni; i++) {
          data[i_elem++] = j * 2.5;
        }
      }
    }
    assert(n_hexa_elems == i_elem);
    // common info for all steps:
    auto sol_name_prefix = std::string("CellData");
    char field_name[kNameLength + 1] = "Temperature";
    char sol_names[kNameLength * kSteps + 1];
    char* dest = sol_names;
    // write solution for each step:
    cgsize_t range_min, range_max = 0;
    for (int k = 0; k < kSteps; ++k) {
      int sol_id;
      auto sol_name = sol_name_prefix + std::to_string(k);
      cg_sol_write(file_id, base_id, zone_id,
          sol_name.c_str(), CGNS_ENUMV(CellCenter), &sol_id);
      std::strcpy(dest, sol_name.c_str());
      dest += kNameLength;
      // partial write:
      int field_id;
      range_min = range_max + 1;
      range_max += (nj-1) * (ni-1);
      cg_field_partial_write(file_id, base_id, zone_id, sol_id,
          CGNS_ENUMV(RealDouble), field_name,
          &range_min, &range_max, &data[range_min-1], &field_id);
    }
    // create BaseIterativeData_t:
    cg_biter_write(file_id, base_id, "TimeIterValues", kSteps);
    // goto this BaseIterativeData_t:
    cg_goto(file_id, base_id, "BaseIterativeData_t", 1, "end");
    // write time values:
    {
      cgsize_t data_dim[1] = {kSteps};  // data size in each dimension
      cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, data_dim, &time);
    }
    // create ZoneIterativeData_t:
    cg_ziter_write(file_id, base_id, zone_id, "ZoneIterativeData");
    // goto this ZoneIterativeData_t:
    cg_goto(file_id, base_id,
        "Zone_t", zone_id, "ZoneIterativeData_t", 1, "end");
    // define which flow FlowSolution_t corresponds with which time step:
    {
      cgsize_t data_dim[2] = {kNameLength, kSteps};
      cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character),
          2, data_dim, sol_names);
    }
    // add SimulationType:
    cg_simulation_type_write(file_id, base_id, CGNS_ENUMV(TimeAccurate));
  }
  /*
    Close the CGNS File
   */
  std::printf("Closing the CGNS file...\n");
  cg_close(file_id);
  std::printf("Successfully wrote unstructured grid to \"%s\".\n", file_name);
  return 0;
}
