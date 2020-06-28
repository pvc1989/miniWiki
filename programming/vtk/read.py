import vtk

reader = vtk.vtkDataSetReader()
reader.SetFileName("ugrid_demo_binary.vtk")
reader.Update()
grid = reader.GetOutput()
# 遍历 Points
for p in range(grid.GetNumberOfPoints()):
  x, y, z = grid.GetPoint(p)
  print(p, ':', x, y, z)
# 遍历 Cells
for c in range(grid.GetNumberOfCells()):
  print(c, ':', end=' ')
  cell = grid.GetCell(c)
  for p in range(cell.GetNumberOfPoints()):
    print(cell.GetPointId(p), end=' ')
  print()

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("ugrid_demo_ascii.vtu")
reader.Update()
grid = reader.GetOutput()
# 遍历 Points
for p in range(grid.GetNumberOfPoints()):
  x, y, z = grid.GetPoint(p)
  print(p, ':', x, y, z)
# 遍历 Cells
for c in range(grid.GetNumberOfCells()):
  print(c, ':', end=' ')
  cell = grid.GetCell(c)
  for p in range(cell.GetNumberOfPoints()):
    print(cell.GetPointId(p), end=' ')
  print()
