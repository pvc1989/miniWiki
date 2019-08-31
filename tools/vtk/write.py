import vtk

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
xyz =((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
      (2, 0, 0), (2, 1, 0), (2, 2, 0), (0, 2, 0))

# 创建「点集 (vtkPoints)」
points = vtk.vtkPoints()
points.SetNumberOfPoints(len(xyz))
# 创建「数组 (vtkFloatArray)」
vector_on_points = vtk.vtkFloatArray()
# virtual void vtkAbstractArray::SetName(const char*)	
vector_on_points.SetName("Position")
vector_on_points.SetNumberOfComponents(3)
vector_on_points.SetNumberOfTuples(len(xyz))
for i in range(len(xyz)):
  points.InsertPoint(i, xyz[i])
  vector_on_points.InsertTuple(i, xyz[i])
# virtual void vtkPointSet::SetPoints(vtkPoints*)
grid.SetPoints(points)
# vtkPointData* vtkDataSet::GetPointData()
data_on_points = grid.GetPointData()
# int vtkDataSetAttributes::SetVectors(vtkDataArray*)
data_on_points.SetVectors(vector_on_points)

# void vtkUnstructuredGrid::Allocate(vtkIdType numCells=1000)
grid.Allocate(4)
# vtkIdList* vtkCell::GetPointIds()
# vtkIdList::SetId(const vtkIdType i_local, const vtkIdType i_global)
# vtkUnstructuredGridBase::InsertNextCell(int type, vtkIdList* id_list)
# 添加两个「三角形单元 (vtkTriangle)」
for global_id_list in ((0, 2, 3), (2, 1, 0)):
  cell = vtk.vtkTriangle()
  id_list = cell.GetPointIds()
  for i in range(3):
    id_list.SetId(i, global_id_list[i])
  grid.InsertNextCell(cell.GetCellType(), id_list)
# 添加两个「四边形单元 (vtkQuad)」
for global_id_list in ((1, 4, 5, 2), (5, 6, 7, 3)):
  cell = vtk.vtkQuad()
  id_list = cell.GetPointIds()
  for i in range(4):
    id_list.SetId(i, global_id_list[i])
  grid.InsertNextCell(cell.GetCellType(), id_list)

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
