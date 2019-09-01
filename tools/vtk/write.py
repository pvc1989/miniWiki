import vtk
import numpy as np

# 创建「非结构网格 (vtkUnstructuredGrid)」
# vtkDataSet <- vtkPointSet <- vtkUnstructuredGridBase <- vtkUnstructuredGrid
grid = vtk.vtkUnstructuredGrid()

# 7 ------------- 6
# |               |
# |      (3)      |
# |               |
# 3 ----- 2 ----- 5
# | (0) / |       |
# |   /   |  (2)  |
# | / (1) |       |
# 0 ----- 1 ----- 4
xyz = np.array([
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [2, 0, 0], [2, 1, 0], [2, 2, 0], [0, 2, 0]], dtype=float)
n_points = len(xyz)
connectivity = ((0, 2, 3), (2, 1, 0), (1, 4, 5, 2), (5, 6, 7, 3))
n_cells = len(connectivity)

# 创建「点集 (vtkPoints)」
vtk_points = vtk.vtkPoints()
vtk_points.SetNumberOfPoints(n_points)
# 创建「数组 (vtkFloatArray)」
vector_on_points = vtk.vtkFloatArray()
vector_on_points.SetName("Position")
vector_on_points.SetNumberOfComponents(3)
vector_on_points.SetNumberOfTuples(n_points)
for i in range(n_points):
  vtk_points.InsertPoint(i, xyz[i])
  vector_on_points.InsertTuple(i, xyz[i])
# vtkPointSet::SetPoints(vtkPoints*)
grid.SetPoints(vtk_points)
# vtkPointData* vtkDataSet::GetPointData()
# vtkDataSetAttributes::SetVectors(vtkDataArray*)
# vtkDataSetAttributes <- vtkPointData
grid.GetPointData().SetVectors(vector_on_points)

# 创建四个「单元 (vtkCell)」
# void vtkUnstructuredGrid::Allocate(vtkIdType numCells=1000)
grid.Allocate(n_cells)
vector_on_cells = vtk.vtkFloatArray()
vector_on_cells.SetName("Average Position")
vector_on_cells.SetNumberOfComponents(3)
vector_on_cells.SetNumberOfTuples(n_cells)
for i_cell in range(n_cells):
  global_id_list = connectivity[i_cell]
  n_nodes = len(global_id_list)
  if n_nodes == 3:
    vtk_cell = vtk.vtkTriangle()  # vtkCell <- vtkTriangle
  elif n_nodes == 4:
    vtk_cell = vtk.vtkQuad()      # vtkCell <- vtkQuad
  else:
    assert(False)
  vtk_id_list = vtk_cell.GetPointIds()  # vtkIdList* vtkCell::GetPointIds()
  xyz_average = np.zeros(3)
  for i in range(n_nodes):
    # vtkIdList::SetId(const vtkIdType i_local, const vtkIdType i_global)
    i_node = global_id_list[i]
    vtk_id_list.SetId(i, i_node)
    xyz_average += xyz[i_node]
  # vtkUnstructuredGridBase::InsertNextCell(int type, vtkIdList* vtk_id_list)
  grid.InsertNextCell(vtk_cell.GetCellType(), vtk_id_list)
  # 以结点坐标的平均值为「单元数据 (cell data)」
  xyz_average /= n_nodes
  vector_on_cells.InsertTuple(i_cell, xyz_average)
# vtkCellData* vtkDataSet::GetCellData()
# vtkDataSetAttributes::SetVectors(vtkDataArray*)
# vtkDataSetAttributes <- vtkCellData
grid.GetCellData().SetVectors(vector_on_cells)

# 按 传统 VTK 格式 输出
writer = vtk.vtkDataSetWriter()
writer.SetInputData(grid)
# 输出为 纯文本 文件
writer.SetFileName("ugrid_demo_ascii.vtk")
writer.SetFileTypeToASCII()
writer.Write()
# 输出为 二进制 文件
writer.SetFileName("ugrid_demo_binary.vtk")
writer.SetFileTypeToBinary()
writer.Write()

# 按 现代 XML 格式 输出
writer = vtk.vtkXMLDataSetWriter()
writer.SetInputData(grid)
# 输出为 纯文本 文件
writer.SetFileName("ugrid_demo_ascii.vtu")
writer.SetDataModeToAscii()
writer.Write()
# 输出为 二进制 文件
writer.SetFileName("ugrid_demo_binary.vtu")
writer.SetDataModeToBinary()
writer.Write()
