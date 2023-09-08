---
title: Plot3D
---

# Data Format

```python
 8                                   # Number of blocks
    5       5       5                # Block 1 is shaped 5x5x5 in i,j,k this is 125 nodes
    5       5       5                # Block 2 shape
 1.08569236018e-07  2.89960779720e-07  2.99150448968e-07  2.55513369907e-07          # First 125 values describe are X, next 25 are Y, last 25 are Z
 2.12244548714e-07  1.01483086157e-07  2.67218012828e-07  2.71062573276e-07
 2.24420893535e-07  1.78210401103e-07  7.47886754748e-08  1.93926065872e-07
 1.84719158858e-07  1.38925249638e-07  1.00523800506e-07  1.71518139691e-08
 9.37154616132e-08  9.36074304736e-08  5.80944219397e-08  2.00380892990e-08
 -5.46866818496e-08 -1.24404013757e-08  1.15859071226e-08 -4.76059469623e-09
```

⚠️ **NPARC Alliance Validation Archive > [Validation Home](https://www.grc.nasa.gov/WWW/wind/valid/homepage.html) > [Archive](https://www.grc.nasa.gov/WWW/wind/valid/archive.html) > [ONERA M6 Wing](https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html) > [Study #1](https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing01/m6wing01.html)** 提供的 [`m6wing.x.fmt`](https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing01/m6wing.x.fmt) 文件与上述格式有一些差别。以下 Python 函数将其修复为符合上述格式的 `m6wing-ascii.xyz` 文件：

```python
def fmt_to_ascii_xyz():
    n_val = 0
    n_line = 0
    n_block = 0
    with open('m6wing-ascii.xyz', 'w') as new_file:
        with open('m6wing.x.fmt') as old_file:
            for old_line in old_file:
                n_line += 1
                if n_line == 1:
                    n_block = int(old_line)
                    new_file.write(old_line.strip())
                    new_file.write('\n')
                    continue
                words = old_line.split(',')
                if n_line == 2:
                    for i_block in range(n_block):
                        for i in range(3 * i_block, 3 * (i_block + 1)):
                            new_file.write(words[i].strip())
                            new_file.write(' ')
                        new_file.write('\n')
                    continue
                for word in words:
                    word = word.strip()
                    if len(word) == 0:
                        continue
                    n_star = word.count('*')
                    if n_star == 1:
                        cnt, val = word.split('*')
                        cnt = int(cnt)
                        for i in range(cnt):
                            n_val += 1
                            new_file.write(val)
                            new_file.write(' ')
                    else:
                        assert n_star == 0
                        n_val += 1
                        new_file.write(word)
                        new_file.write(' ')
                new_file.write('\n')
    print('n_value =', n_val, 'n_point =', n_val // 3)
```

# Python Module

- [GitHub](https://github.com/nasa/Plot3D_utilities)
- [Documentation](https://nasa.github.io/Plot3D_utilities/_build/html)

```shell
pip install plot3d
```

```python
import plot3d

def ascii_to_binary():
    blocks = plot3d.read_plot3D('m6wing-ascii.xyz', binary=False)
    print(blocks)
    plot3d.write_plot3D('m6wing.xyz', blocks, binary=True)
    print(plot3d.get_outer_bounds(blocks))
```

## To [VTK](./vtk.md)

```python
from vtk import vtkMultiBlockPLOT3DReader
from vtk import vtkXMLMultiBlockDataReader, vtkXMLMultiBlockDataWriter
from vtk import vtkMultiBlockDataSet, vtkStructuredGrid, vtkPoints

def xyz_to_vtm():
    reader = vtkMultiBlockPLOT3DReader()
    reader.SetBinaryFile(True)
    reader.SetFileName('m6wing.xyz')
    reader.Update()
    grid = reader.GetOutput()
    print(grid)

    writer = vtkXMLMultiBlockDataWriter()
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    writer.SetFileName('m6wing.vtm')
    writer.Write()
```

## To [CGNS](./cgns.md)

```python
from vtk import vtkMultiBlockPLOT3DReader
from vtk import vtkXMLMultiBlockDataReader, vtkXMLMultiBlockDataWriter
from vtk import vtkMultiBlockDataSet, vtkStructuredGrid, vtkPoints
import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnskeywords as cgk
import numpy as np

def vtm_to_cgns():
    tree = cgl.newCGNSTree(3.4)
    print(tree)
    base = cgl.newBase(tree, 'M6_Wing', 3, 3) # name, physical dim, topological dim
    print("  ", base)
    reader = vtkXMLMultiBlockDataReader()
    reader.SetFileName('m6wing.vtm')
    reader.Update()
    grid = reader.GetOutput()
    assert isinstance(grid, vtkMultiBlockDataSet)
    n_zone = grid.GetNumberOfBlocks()
    for i_zone in range(n_zone):
        block = grid.GetBlock(i_zone)
        assert isinstance(block, vtkStructuredGrid)
        imax, jmax, kmax = block.GetDimensions()
        zone_size = np.array([
            [imax, imax-1, 0],
            [jmax, jmax-1, 0],
            [kmax, kmax-1, 0]], order='F')
        zone_type = cgk.Structured_s
        zone = cgl.newZone(base, f'Zone{i_zone}', zone_size, zone_type)
        print("    ", zone)
        points = block.GetPoints()
        assert isinstance(points, vtkPoints)
        n_point = block.GetNumberOfPoints()
        assert n_point == imax * jmax * kmax
        x = np.ndarray((imax, jmax, kmax))
        y = np.ndarray((imax, jmax, kmax))
        z = np.ndarray((imax, jmax, kmax))
        i_point = 0
        for i in range(imax):
            for j in range(jmax):
                for k in range(kmax):
                    point_i = points.GetPoint(i_point)
                    x[i][j][k] = point_i[0]
                    y[i][j][k] = point_i[1]
                    z[i][j][k] = point_i[2]
                    i_point += 1
        assert i_point == n_point
        cgl.newCoordinates(zone, cgk.CoordinateX_s, x)
        cgl.newCoordinates(zone, cgk.CoordinateY_s, y)
        cgl.newCoordinates(zone, cgk.CoordinateZ_s, z)
    cgm.save('m5wing.cgns', tree)
```
