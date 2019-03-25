import vtk

# 创建 vtkPoints (点集) 对象
points = vtk.vtkPoints()
points.SetNumberOfPoints(6)
# void vtkPoints::InsertPoint(vtkIdType id, double x, double y, double z)
points.InsertPoint(0, 0, 0, 0)
points.InsertPoint(1, 1, 0, 0)
points.InsertPoint(2, 1, 1, 0)
points.InsertPoint(3, 0, 1, 0)
points.InsertPoint(4, 2, 0, 0)
points.InsertPoint(5, 2, 1, 0)

# 创建 vtkUnstructuredGrid (非结构网格) 对象
grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(points)

# void vtkUnstructuredGrid::Allocate(vtkIdType numCells=1000, int extSize=1000)
# extSize 现已弃用, 不需要设置
grid.Allocate(3)

# 创建 vtkQuad (四边形单元) 对象
cell = vtk.vtkQuad()
# 定义 局部结点编号 到 全局结点编号 的映射
# vtkIdList* vtkCell::GetPointIds ()
idList = cell.GetPointIds()
# void vtkIdList::SetId(const vtkIdType i, const vtkIdType vtkid)
idList.SetId(0, 1)
idList.SetId(1, 4)
idList.SetId(2, 5)
idList.SetId(3, 2)
# 将单元添加进 grid 中
# vtkIdType vtkUnstructuredGridBase::InsertNextCell(int type, vtkIdList* ptIds)
grid.InsertNextCell(cell.GetCellType(), idList)

# 创建第 1 个 vtkTriangle (三角形单元) 对象, 并添加进 grid 中
cell = vtk.vtkTriangle()
idList = cell.GetPointIds()
idList.SetId(0, 0)
idList.SetId(1, 2)
idList.SetId(2, 3)
grid.InsertNextCell(cell.GetCellType(), idList)
# 创建第 2 个 vtkTriangle (三角形单元) 对象, 并添加进 grid 中
cell = vtk.vtkTriangle()
idList = cell.GetPointIds()
idList.SetId(0, 2)
idList.SetId(1, 1)
idList.SetId(2, 0)
grid.InsertNextCell(cell.GetCellType(), idList)

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
