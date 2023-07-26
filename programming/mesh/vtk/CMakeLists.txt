cmake_minimum_required(VERSION 3.3...3.12 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

PROJECT(IO)

find_package(VTK REQUIRED)
if (vtkIOExport_LOADED AND vtkIOImport_LOADED)
vtk_module_config(VTK
  vtkCommonCore
  vtkIOLegacy
  vtkIOXML
  vtkIOGeometry
  vtkIOImport
  vtkIOExport
  vtksys
)
else()
vtk_module_config(VTK
  vtkCommonCore
  vtkIOLegacy
  vtkIOXML
  vtkIOGeometry
  vtkIOImport
  vtksys
)
endif()
include(${VTK_USE_FILE})

add_executable(read read.cpp)
target_link_libraries(read ${VTK_LIBRARIES})
