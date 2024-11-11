---
title: OCC (Open CASCADE)
---

# [OCCT (Open CASCADE Technology)](https://dev.opencascade.org)

- [GitHub](https://github.com/Open-Cascade-SAS/OCCT)
  - [User Guides](https://github.com/Open-Cascade-SAS/OCCT/wiki/user_guides)
    - [STEP Translator](https://github.com/Open-Cascade-SAS/OCCT/wiki/step)
- [Documentation](https://dev.opencascade.org/doc/overview/html/index.html)

# [pythonocc-core](https://github.com/tpaviot/pythonocc-core)

## [Install with `conda`](https://github.com/tpaviot/pythonocc-core?tab=readme-ov-file#install-with-conda)

安装 `conda`：

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# 根据提示，输入 yes 或 ENTER
# 安装完成后，需重启 shell，以更新环境变量
```

安装 `pythonocc`：

```shell
conda create --name=pyoccenv
source activate pyoccenv
conda install -c conda-forge pythonocc-core
```

## 典型用例

### 读取 STEP 文件

```python
from OCC.Core.STEPControl import STEPControl_Reader

def get_one_shape_from_cad(step_file: str):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)
    if status != 1:
        raise Exception("Error reading STEP file.")
    step_reader.TransferRoots()
    one_shape = step_reader.OneShape()
    return one_shape
```

### 查询最近点

```python
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.gp import gp_Pnt

def get_nearest_point(x: float, y: float, z: float, shape: TopoDS_Shape) -> tuple[float, float, float]:
    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
    dist_shape_shape = BRepExtrema_DistShapeShape(vertex, shape)
    dist_shape_shape.Perform()
    point_on_shape = dist_shape_shape.PointOnShape2(1)
    distance = dist_shape_shape.Value()
    x, y, z = point_on_shape.X(), point_on_shape.Y(), point_on_shape.Z()
    return x, y, z, distance
```
