/*
  Read VTK files, and report their contents.
  Usage: read file_1 file_2 ...
      where `file_x` is a file with extension `.vtu` or `.vtk`.
 */
// For .vtk files:
#include <vtkDataSetReader.h>
#include <vtkDataSet.h>
// For .vtu files:
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
// DataAttributes:
#include <vtkFieldData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
// Helps:
#include <vtkSmartPointer.h>
#include <vtkCellTypes.h>
#include <vtksys/SystemTools.hxx>
// STL:
#include <map>

template <class Reader>
vtkDataSet* Read(const char* file_name) {
  auto reader = vtkSmartPointer<Reader>::New();
  reader->SetFileName(file_name);
  reader->Update();
  reader->GetOutput()->Register(reader);
  return vtkDataSet::SafeDownCast(reader->GetOutput());
}

vtkDataSet* Dispatch(const char* file_name) {
  vtkDataSet* data_set{nullptr};
  auto extension = vtksys::SystemTools::GetFilenameLastExtension(file_name);
  // Dispatch based on the file extension
  if (extension == ".vtu") {
    data_set = Read<vtkXMLUnstructuredGridReader>(file_name);
  }
  else if (extension == ".vtk") {
    data_set = Read<vtkDataSetReader>(file_name);
  }
  else {
    std::cerr << "Unknown extension: " << extension << std::endl;
  }
  return data_set;
}

void Check(vtkFieldData* field_data, const char* field) {
  if (field_data) {
    int n_arrays = field_data->GetNumberOfArrays();
    std::cout << "  contains " << field << " data with "
              << n_arrays << " arrays.\n";
    for (int i = 0; i < n_arrays; i++) {
      auto array_name = field_data->GetArrayName(i);
      std::cout << "    Array " << i << " is named "
                << (array_name ? array_name : "NULL") << ".\n";
    }
  }
}

void Report(const char* file_name, vtkDataSet* data_set) {
  int n_cells = data_set->GetNumberOfCells();
  int n_points = data_set->GetNumberOfPoints();
  // Generate a report
  std::cout << "------------------------" << std::endl;
  std::cout << file_name << "\n  contains a " << data_set->GetClassName()
        << " that has " << n_cells << " cells"
        << " and " << n_points << " points." << std::endl;
  auto type_to_count = std::map<int, int>();
  for (int i = 0; i < n_cells; i++) {
    type_to_count[data_set->GetCellType(i)]++;
  }
  for (auto [type, count] : type_to_count) {
    std::cout << "    Cell type " << vtkCellTypes::GetClassNameFromTypeId(type)
              << " occurs " << count << " times." << std::endl;
  }
}

int Process(const char* file_name) {
  auto data_set = Dispatch(file_name);  // Do NOT forget data_set->Delete().
  if (data_set == nullptr) { return EXIT_FAILURE; }
  Report(file_name, data_set);
  Check(data_set->GetPointData(), "point");
  Check(data_set->GetCellData(), "cell");
  data_set->Delete();
  return EXIT_SUCCESS;
}

int main (int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " file_1 file_2 ..." << std::endl;
  }
  for (int i = 1; i != argc; ++i) {
    if (Process(argv[i]) == EXIT_FAILURE) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
