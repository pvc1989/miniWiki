---
title: 复变函数论、复分析
---

# 复数

## 定义

### 有序实数对

$$
\mathbb{C}\coloneqq\left\{ (x,y):(x,y)\in\mathbb{R}^{2}\right\} 
$$

$$
(x_{1},y_{1})+(x_{2},y_{2})\coloneqq(x_{1}+x_{2},\ y_{1}+y_{2})
$$

$$
(x_{1},y_{1})\times(x_{2},y_{2})\coloneqq(x_{1}x_{2}-y_{1}y_{2},\ x_{1}y_{2}+x_{2}y_{1})
$$

$$
1\equiv(1,0),\quad
\sqrt{-1}\equiv(0,1)
$$

### 反对称实数矩阵

$$
\mathbb{C}\coloneqq\left\{ \begin{bmatrix}x & y\\
-y & x
\end{bmatrix}:(x,y)\in\mathbb{R}^{2}\right\} 
$$

$$
\begin{bmatrix}x_{1} & y_{1}\\
-y_{1} & x_{1}
\end{bmatrix}+\begin{bmatrix}x_{2} & y_{2}\\
-y_{2} & x_{2}
\end{bmatrix} \coloneqq\begin{bmatrix}x_{1}+x_{2} & y_{1}+y_{2}\\
-y_{1}-y_{2} & x_{1}+x_{2}
\end{bmatrix}
$$

$$
\begin{bmatrix}x_{1} & y_{1}\\
-y_{1} & x_{1}
\end{bmatrix}\times\begin{bmatrix}x_{2} & y_{2}\\
-y_{2} & x_{2}
\end{bmatrix} \coloneqq\begin{bmatrix}x_{1}x_{2}-y_{1}y_{2} & x_{1}y_{2}+x_{2}y_{1}\\
-x_{1}y_{2}-x_{2}y_{1} & x_{1}x_{2}-y_{1}y_{2}
\end{bmatrix}
$$

$$
1\equiv\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix},\quad
\sqrt{-1}\equiv\begin{bmatrix}0 & 1\\-1 & 0\end{bmatrix}
$$

## 几何表示

### 直角坐标

$$
z=x+\sqrt{-1}y,\quad(x,y)\in\mathbb{R}^{2}
$$

- 实部 $$ \mathrm{Re}(z)\coloneqq x $$
- 虚部 $$ \mathrm{Im}(z)\coloneqq y $$
- 复共轭 $$ z^* \coloneqq x - \sqrt{-1}y $$

### 极坐标

$$
z=\rho\cos\theta+\sqrt{-1}\rho\sin\theta,\quad(\rho,\theta)\in[0,\infty)\times(-\infty,\infty)
$$

- 模 $$ |z|\equiv\rho\coloneqq\sqrt{zz^*} $$
- 辐角 $$ \theta \coloneqq \arctan(y/x) $$
- 辐角主值 $$ \theta\in[0,2\mathrm{\pi}) $$

### 复平面

复平面默认不含无穷远点，即

$$
\mathbb{C}\coloneqq\left\{ x+\sqrt{-1}y:\vert x\vert+\vert y\vert<\infty\right\} 
$$

### 复球面

以复平面为赤道面（或与复平面原点相切）的球面。

在复球面上，$$ \forall\theta $$，极限 $$ \lim_{\substack{\rho\to\infty\\\rho\in\mathbb{R}}}\rho\exp(\sqrt{-1}\theta) $$ 趋向同一点，故复球面可记作

$$
\overline{\mathbb{C}}\coloneqq\mathbb{C}\cup\left\{ \infty\right\} 
$$

## 复数序列

基于复数的模，可以定义距离

$$
d(p,q)\coloneqq\vert p-q\vert,\quad\forall(p,q)\in\mathbb{C}^{2}.
&&

### 序列的极限

$$\forall \varepsilon>0,\exists N>0,\forall n > N,\abs{z_n - a} < \epsilon$$

### Cauchy 收敛准则

若 $$\forall \varepsilon>0,\exists N>0,\forall m > N \land n > N,\abs{z_m - z_n} < \epsilon$$，则该序列收敛。

### 序列的聚点<a href id="limit_point"></a>

若某序列的无穷子列收敛，则该子列的极限是原序列的一个***聚点***，又称***极限点***。

注. 一个复数序列可以由多个*聚点*，但至多只能有一个*极限*。

## 点集拓扑

基于上述距离定义，可导出 $$ \mathbb{C} $$ 或 $$ \overline{\mathbb{C}} $$ 上的拓扑。

### 点与点集的关系

给定点 $$ z $$ 与点集 $$ D $$：

- 邻域：
- 内点：存在 $$ z $$ 的充分小邻域，其所有成员点都 $$ \in D $$。
- 外点：存在 $$ z $$ 的充分小邻域，其所有成员点都 $$ \notin D $$。
- 边界点：在 $$ z $$ 的任意邻域内，都既有内点、又有外点。

### 点集的分类

给定点集 $$ D $$：

- 开集：$$ D $$ 的所有成员点都是 $$ D $$ 的内点。
- 闭集：$$ D $$ 是某个开集的补集。
- 闭包：$$ D $$ 与其所有[聚点](#limit_point)之集的并。
- 连通性：$$ D $$ 内任意两点都可通过 $$ D $$ 的成员点相连。
  - 单连通：所有闭合曲线都可以收缩到一点。
  - 复连通：非单连通的连通集。
- 区域：连通的开集。

## 复数级数

# 复变函数

## 极限

## 连续性

## 函数项级数

## 多值函数

# 微分学

## 可导性

## 解析性

## 保角变换

# 积分学

## 路径积分

## Cauchy 定理

## Cauchy 积分公式

## Cauchy 型积分

## 特殊函数

# 幂级数

## Taylor 级数

## Laurent 级数

# 留数定理

## 留数

## 定理

## 定积分

## 其他应用
