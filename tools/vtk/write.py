import vtk

# 3 ----- 2 ----- 5
# | (1) / |       |
# |   /   |  (0)  |
# | / (2) |       |
# 0 ----- 1 ----- 4
xyz =((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0))

# 创建 vtkPoints (点集) 对象
points = vtk.vtkPoints()
points.SetNumberOfPoints(len(xyz))
# 创建 vtkFloatArray (float 型数组) 对象
vector_on_points = vtk.vtkFloatArray()
# virtual void vtkAbstractArray::SetName(const char* name)	
vector_on_points.SetName("Position")
vector_on_points.SetNumberOfComponents(3)
vector_on_points.SetNumberOfTuples(len(xyz))
for i in range(6):
  points.InsertPoint(i, xyz[i])
  vector_on_points.InsertTuple(i, xyz[i])

# 创建 vtkUnstructuredGrid (非结构网格) 对象
grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(points)
# vtkPointData* vtkDataSet::GetPointData()
data_on_points = grid.GetPointData()
# int vtkDataSetAttributes::SetVectors(vtkDataArray* da)
data_on_points.SetVectors(vector_on_points)

# void vtkUnstructuredGrid::Allocate(vtkIdType numCells=1000, int extSize=1000)
# extSize 现已弃用, 不需要设置
grid.Allocate(3)

# 创建 vtkQuad (四边形单元) 对象
cell = vtk.vtkQuad()
# 定义 局部结点编号 到 全局结点编号 的映射
# vtkIdList* vtkCell::GetPointIds ()
id_list = cell.GetPointIds()
# void vtkIdList::SetId(const vtkIdType i, const vtkIdType vtkid)
id_list.SetId(0, 1)
id_list.SetId(1, 4)
id_list.SetId(2, 5)
id_list.SetId(3, 2)
# 将单元添加进 grid 中
# vtkIdType vtkUnstructuredGridBase::InsertNextCell(int type, vtkIdList* ptIds)
grid.InsertNextCell(cell.GetCellType(), id_list)

# 创建第 1 个 vtkTriangle (三角形单元) 对象, 并添加进 grid 中
cell = vtk.vtkTriangle()
id_list = cell.GetPointIds()
id_list.SetId(0, 0)
id_list.SetId(1, 2)
id_list.SetId(2, 3)
grid.InsertNextCell(cell.GetCellType(), id_list)
# 创建第 2 个 vtkTriangle (三角形单元) 对象, 并添加进 grid 中
cell = vtk.vtkTriangle()
id_list = cell.GetPointIds()
id_list.SetId(0, 2)
id_list.SetId(1, 1)
id_list.SetId(2, 0)
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
