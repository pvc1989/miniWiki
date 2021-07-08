---
title: 流体力学（理论与计算）
---

# 流动方程

## 速度导数

### 速度梯度

**速度微分 (differential of velocity)** 可以表示为**矢径微分 (differential of position vector)** 与**速度梯度 (gradient of velocity)** 的点乘之积：
$$
\dd{\Vec{u}}=\dd{\Vec{r}}\vdot\grad\Vec{u}
$$

其中*速度梯度*是一个二阶张量：

$$
\grad\Vec{u}\coloneqq\left(\Vec{e}_{i}\pdv{}{x_{i}}\right)\left(\Vec{e}_{k}u_{k}\right)=\left(\Vec{e}_{i}\Vec{e}_{k}\right)\pdv{u_{k}}{x_{i}}\eqqcolon\Vec{e}_{i}\Vec{e}_{k}\partial_{i}u_{k}
$$

它可以分解为**应变率张量 (tensor of strain rate)** 与**角速度张量 (tensor of angular velocity)** 之和：

$$
\boxed{\grad\Vec{u}=\VecVec{E}+\VecVec{\varOmega}}\impliedby\begin{cases}
\VecVec{E}\coloneqq\Vec{e}_{i}\Vec{e}_{k}E_{ik} & E_{ik}\coloneqq(\partial_{i}u_{k}+\partial_{k}u_{i})/2\\
\VecVec{\varOmega}\coloneqq\Vec{e}_{i}\Vec{e}_{k}\varOmega_{ik} & \varOmega_{ik}\coloneqq(\partial_{i}u_{k}-\partial_{k}u_{i})/2
\end{cases}
$$

### 角速度

反对称的*角速度张量*只有三个独立分量：

$$
\VecVec{\varOmega}=\begin{bmatrix}\Vec{e}_{1} & \Vec{e}_{2} & \Vec{e}_{3}\end{bmatrix}\begin{bmatrix}0 & +\varOmega_{12} & -\varOmega_{31}\\
-\varOmega_{12} & 0 & +\varOmega_{23}\\
+\varOmega_{31} & -\varOmega_{23} & 0
\end{bmatrix}\begin{bmatrix}\Vec{e}_{1}\\
\Vec{e}_{2}\\
\Vec{e}_{3}
\end{bmatrix}
$$

利用这三个独立分量，可以人为构造一个三维**角速度矢量 (vector of angular velocity)**：

$$
\begin{bmatrix}\varOmega_{1}\\
\varOmega_{2}\\
\varOmega_{3}
\end{bmatrix}=\begin{bmatrix}\varOmega_{23}\\
\varOmega_{31}\\
\varOmega_{12}
\end{bmatrix}\iff\varOmega_{jk}=\varOmega_{i}\epsilon_{ijk}\quad\text{其中}\quad\epsilon_{ijk}=\begin{cases}
+1 & ijk=123,231,312\\
-1 & ijk=321,132,213\\
0  & ijk=\cdots
\end{cases}
$$

可以证明：

$$
\boxed{\dd{\Vec{r}}\vdot\VecVec{\varOmega}=\Vec{\varOmega}\cross\dd{\Vec{r}}}\impliedby\begin{cases}
\Vec{\varOmega}\coloneqq\Vec{e}_{i}\varOmega_{i}\\
\VecVec{\varOmega}\coloneqq\Vec{e}_{j}\Vec{e}_{k}\varOmega_{jk}=\varOmega_{i}\Vec{e}_{j}\Vec{e}_{k}\epsilon_{ijk}
\end{cases}
$$

### 涡量

$$
\Vec{\omega}\coloneqq\curl{\Vec{u}}=\begin{vmatrix}\Vec{e}_{1} & \Vec{e}_{2} & \Vec{e}_{3}\\
\partial_{1} & \partial_{2} & \partial_{3}\\
u_{1} & u_{2} & u_{3}
\end{vmatrix}=2\Vec{\varOmega}
$$

## 物质导数

### Reynolds 输运定理

$$
\boxed{\dv{}{t}\int_{V}\phi=\int_{V}\pdv{\phi}{t}+\oint_{\partial V}\Vec{n}\vdot\Vec{u}_{\mathrm{C.S.}}\,\phi}
$$

### 控制体上的物质导数

$$
\boxed{\begin{aligned}\dv{}{t}\int_{V_{\mathrm{M.B.}}}\phi\eqqcolon\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\phi & =\int_{V}\pdv{\phi}{t}+\oint_{\partial V}\Vec{n}\vdot\Vec{u}\phi\\
 & =\dv{}{t}\int_{V}\phi+\oint_{\partial V}\Vec{n}\vdot\left(\Vec{u}-\Vec{u}_{\mathrm{C.S.}}\right)\phi
\end{aligned}
}
$$

### 物质点上的物质导数

$$
\boxed{\frac{\mathrm{D}\phi}{\mathrm{D}t}\equiv\mathrm{D}_{t}{\phi}\coloneqq\partial_{t}\phi+\Vec{u}\vdot\grad\phi}
$$

## 守恒定律

### 质量守恒

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho=0}
$$

|          |                            积分型                            |                         微分型                          |
| :------: | :----------------------------------------------------------: | :-----------------------------------------------------: |
|  守恒型  | $\int_{V}\pdv{\rho}{t}+\oint_{\partial V}\Vec{n}\vdot\Vec{u}\rho=0$ |          $\pdv{\rho}{t}+\divg(\Vec{u}\rho)=0$           |
| 非守恒型 | $\dv{}{t}\int_{V}\rho+\oint_{\partial V}\Vec{n}\vdot\left(\Vec{u}-\Vec{u}_{\mathrm{C.S.}}\right)\rho=0$ | $\frac{\mathrm{D}\rho}{\mathrm{D}t}+\rho\divg\Vec{u}=0$ |

### 动量守恒

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho\Vec{u}=\int_{V}\Vec{b}+\oint_{\partial V}\Vec{n}\vdot\VecVec{\sigma}}
$$

其中 $\VecVec{\sigma}$ 为**应力张量 (stress tensor)**，它与单位法向量 $\Vec{n}$ 的点乘 $\Vec{n}\vdot\VecVec{\sigma}$ 是一个向量，表示作用在物质体边界 $\partial V_{\mathrm{M.B.}}$ 上的外力的面密度。在流体力学中，应力张量 $\VecVec{\sigma}$ 总是被分解为**静水压力张量 (hydrostatic pressure tensor) $-p\VecVec{1}$** 与**黏性应力张量 (viscous stress tensor) $\VecVec{\tau}$** 之和：

$$
\VecVec{\sigma}=-p\VecVec{1}+\VecVec{\tau}\iff\sigma_{ik}=-p\delta_{ik}+\tau_{ik}
$$

于是动量守恒定律可以改写为

$$
\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho\Vec{u}+\oint_{\partial V}\Vec{n}p=\int_{V}\Vec{b}+\oint_{\partial V}\Vec{n}\vdot\VecVec{\tau}
$$

### 角动量守恒

在（不考虑分布力偶的）流体力学中，它等价于*应力张量*或*黏性应力张量*的对称性：

$$
\boxed{\underbrace{-p\delta_{ik}+\tau_{ik}}_{\sigma_{ik}}=\underbrace{-p\delta_{ki}+\tau_{ki}}_{\sigma_{ki}}}
$$

### 能量守恒

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho e_{0}=\int_{V}\big(\Vec{b}\vdot\Vec{u}+j\big)+\oint_{\partial V}\Vec{n}\vdot\big(\VecVec{\sigma}\vdot\Vec{u}-\Vec{q}\big)}
$$

其中
- 标量 $e_0\coloneqq e+\Vec{u}\vdot\Vec{u}/2$ 为**比总能 (specific total energy)**，表示*单位质量*流体的*内能*与*动能*之和。
- 标量 $j$ 为**热源密度 (density of heat source)**，表示*单位时间*内*单位体积*热源的**热生成量 (heat generation)**。
- 向量 $\Vec{q}$ 为**热通量密度 (density of heat flux)**，表示*单位时间*内穿过*单位面积*的**热流量 (heat flow)**。

## Navier–Stokes 方程组

将守恒定律整理成矩阵（方程组）形式，即得 **Navier–Stokes 方程组**：

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\Mat{U}+\oint_{\partial V}\Vec{n}\vdot\Mat{\Vec{P}}=\oint_{\partial V}\Vec{n}\vdot\Mat{\Vec{G}}+\int_{V}\Mat{H}}
$$

其中

$$
\Mat{U}=\begin{bmatrix}\rho\\
\rho\Vec{u}\\
\rho e_{0}
\end{bmatrix}\qquad\Mat{\Vec{P}}=\begin{bmatrix}\Vec{0}\\
p\VecVec{1}\\
p\Vec{u}
\end{bmatrix}\qquad\Mat{\Vec{G}}=\begin{bmatrix}\Vec{0}\\
\VecVec{\tau}\\
\VecVec{\tau}\vdot\Vec{u}-\Vec{q}
\end{bmatrix}\qquad\Mat{H}=\begin{bmatrix}0\\
\Vec{b}\\
\Vec{b}\vdot\Vec{u}+j
\end{bmatrix}
$$

### 守恒型

利用[控制体上的物质导数](#控制体上的物质导数)的展开式，并引入**通量 (flux)** 矩阵

$$
\Mat{\Vec{F}}\coloneqq\Mat{U}\Vec{u}+\Mat{\Vec{P}}=\begin{bmatrix}\rho\\
\rho\Vec{u}\\
\rho e_{0}
\end{bmatrix}\Vec{u}+\begin{bmatrix}\Vec{0}\\
p\VecVec{1}\\
p\Vec{u}
\end{bmatrix}=\begin{bmatrix}\rho\Vec{u}\\
\rho\Vec{u}\Vec{u}+p\VecVec{1}\\
\rho h_{0}\Vec{u}
\end{bmatrix}
$$

即得**守恒型积分方程组**：

$$
\int_{V}\partial_{t}\Mat{U}+\oint_{\partial V}\Vec{n}\vdot\Mat{\Vec{F}}=\oint_{\partial V}\Vec{n}\vdot\Mat{\Vec{G}}+\int_{V}\Mat{H}
$$

对其中的面积分应用 *Gauss 散度定理*，即得**守恒型微分方程组**：

$$
\partial_{t}\Mat{U}+\divg\Mat{\Vec{F}}=\divg\Mat{\Vec{G}}+\Mat{H}
$$

它在三维直角坐标系中的分量形式为

$$
\partial_{t}\mathopen{}\begin{bmatrix}\rho\\
\rho u_{1}\\
\rho u_{2}\\
\rho u_{3}\\
\rho e_{0}
\end{bmatrix}+\partial_{\alpha}\mathopen{}\begin{bmatrix}\rho u_{\alpha}\\
\rho u_{1}u_{\alpha}+p\delta_{1\alpha}\\
\rho u_{2}u_{\alpha}+p\delta_{2\alpha}\\
\rho u_{3}u_{\alpha}+p\delta_{3\alpha}\\
\rho h_{0}u_{\alpha}
\end{bmatrix}=\partial_{\alpha}\mathopen{}\begin{bmatrix}0\\
\tau_{\alpha1}\\
\tau_{\alpha2}\\
\tau_{\alpha3}\\
\tau_{\alpha\beta}u_{\beta}-q_{\alpha}
\end{bmatrix}+\begin{bmatrix}0\\
b_{1}\\
b_{2}\\
b_{3}\\
b_{\alpha}u_{\alpha}+j
\end{bmatrix}
$$

### 非守恒型

注意到守恒项 $\Mat{U}$ 中的 $\rho$ 可以被提出：

$$
\Mat{U}=\rho\Mat{W} \impliedby \Mat{W}\coloneqq\begin{bmatrix}1 \\ \vec{u} \\ e_{0}\end{bmatrix}
$$

故[守恒型积分方程组](#守恒型)中的[物质导数](#物质导数)可移入积分号内，由此即得**非守恒型积分方程组**：

$$
\int_{V}\rho\,\overbrace{\left(\partial_{t}+\Vec{u}\vdot\grad\right)}^{\mathrm{D}_t}\Mat{W}+\oint_{\partial V}\vec{n}\vdot\Mat{\Vec{P}}=\oint_{\partial V}\vec{n}\vdot\Mat{\Vec{G}}+\int_{V}\Mat{H}
$$

相应地有**非守恒型微分方程组**：

$$
\left(\rho\partial_{t}+\rho\Vec{u}\vdot\grad\right)\Mat{W}+\divg\Mat{\Vec{P}}=\divg\Mat{\Vec{G}}+\Mat{H}
$$

它在三维直角坐标系中的分量形式为

$$
\left(\rho\partial_{t}+\rho\Vec{u}\vdot\grad\right)
\begin{bmatrix}\rho\\
u_{1}\\
u_{2}\\
u_{3}\\
e_{0}
\end{bmatrix}+\begin{bmatrix}\rho^{2}\partial_{\alpha}u_{\alpha}\\
\partial_{1}p\\
\partial_{2}p\\
\partial_{3}p\\
\partial_{\alpha}(pu_{\alpha})
\end{bmatrix}=\partial_{\alpha}\mathopen{}\begin{bmatrix}0\\
\tau_{\alpha1}\\
\tau_{\alpha2}\\
\tau_{\alpha3}\\
\tau_{\alpha\beta}u_{\beta}-q_{\alpha}
\end{bmatrix}+\begin{bmatrix}0\\
b_{1}\\
b_{2}\\
b_{3}\\
b_{\alpha}u_{\alpha}+j
\end{bmatrix}
$$

# 有限体积

## 间断有限元

### 局部弱形式

守恒律方程（组）的**微分形式**：

$$
\boxed{\partial_{t}U(\Vec{r},t)+\divg\Vec{F}(\Vec{r},t)=0}
$$

可借助于测函数 $V(\Vec{r})$ 改写**积分形式**：

$$
\int_{\varOmega}\left(\partial_{t}U(\Vec{r},t)+\divg\Vec{F}(\Vec{r},t)\right)V(\Vec{r})=0\qquad\forall V(\Vec{r}),\forall\varOmega
$$

分部积分，即得守恒律方程（组）的**弱形式 (weak form)**：

$$
\int_{\varOmega}V(\Vec{r})\,\partial_{t}U(\Vec{r},t)
=\int_{\varOmega}\Vec{F}(\Vec{r},t)\vdot\grad V(\Vec{r})-\oint_{\partial\varOmega}\Vec{\nu}(\Vec{r})\vdot\Vec{F}(\Vec{r},t)\,V(\Vec{r})\qquad\forall V(\Vec{r}),\forall\varOmega
$$

### 正交基函数

选定*两组*线性独立的**基函数 (basis functions)**：

$$
\mathopen{\Mat{\phi}}(\Vec{r})=\begin{bmatrix}\phi_1(\Vec{r})&\cdots&\phi_K(\Vec{r})\end{bmatrix}\qquad
\mathopen{\Mat{\psi}}(\Vec{r})=\begin{bmatrix}\psi_1(\Vec{r})&\cdots&\psi_L(\Vec{r})\end{bmatrix}
$$

在由其张成在**试函数空间 (space of trial functions)** 与**测函数空间 (space of test functions)** 中，分别寻找未知函数 $U(\Vec{r},t)$ 与测函数 $V(\Vec{r})$ 的最优逼近：

$$
U(\Vec{r},t)\approx U^{h}(\Vec{r},t)\coloneqq\mathopen{\Mat{\phi}}(\Vec{r})\cdot\mathopen{\Mat{\hat{U}}}(t)
\qquad
V(\Vec{r})\approx V^{h}(\Vec{r})\coloneqq\mathopen{\Mat{\psi}}(\Vec{r})\cdot\mathopen{\Mat{\hat{V}}}
$$

将上述近似代入弱形式，并利用 $\Mat{V}$ 的任意性，即得一组常微分方程

$$
\boxed{\Mat{A}_{\varOmega}\cdot\dv{}{t}\mathopen{\Mat{\hat{U}}}(t)=
\mathopen{\Mat{B}_{\varOmega}}\big(\mathopen{\Mat{\hat{U}}}(t)\big)-
\mathopen{\Mat{B}_{\partial\varOmega}}\big(\mathopen{\Mat{\hat{U}}}(t)\big)}
$$

其中

$$
\begin{gathered}
\mathopen{\Mat{\hat{U}}}(t)\coloneqq\begin{bmatrix}\hat{U}_{1}(t)\\ \vdots\\ \hat{U}_{K}(t) \end{bmatrix}
\qquad
\Mat{A}_{\varOmega}\coloneqq\begin{bmatrix}\ip{\psi_{1}}{\phi_{1}}_{\varOmega} & \cdots & \ip{\psi_{1}}{\phi_{K}}_{\varOmega}\\
\vdots & \ddots & \vdots\\
\ip{\psi_{L}}{\phi_{1}}_{\varOmega} & \cdots & \ip{\psi_{L}}{\phi_{K}}_{\varOmega}
\end{bmatrix}
\\
\mathopen{\Mat{B}_{\varOmega}}\big(\mathopen{\Mat{\hat{U}}}(t)\big)\coloneqq\langle\mathopen{\grad}\Mat{\psi}\vert\Vec{F}\rangle_{\varOmega}\coloneqq\int_{\varOmega}\begin{bmatrix}\grad\psi_{1}(\Vec{r})\\
\vdots\\
\grad\psi_{L}(\Vec{r})
\end{bmatrix}\vdot\mathopen{\Vec{F}}\left(U^{h}(\Vec{r},t)\right)
\\
\mathopen{\Mat{B}_{\partial\varOmega}}\big(\mathopen{\Mat{\hat{U}}}(t)\big)\coloneqq\langle\Mat{\psi}\vert F_{\nu}\rangle_{\partial\varOmega}\coloneqq\oint_{\partial\varOmega}\begin{bmatrix}\psi_{1}(\Vec{r})\\
\vdots\\
\psi_{L}(\Vec{r})
\end{bmatrix}\mathopen{F_{\nu}}\left(U_{-}^{h}(\Vec{r},t),U_{+}^{h}(\Vec{r},t)\right)
\end{gathered}
$$

将积分区域取为单元及其边界，即得**有限单元法 (finite element methed, FEM)**。

- 若 $(K=L)$ 且 $(\forall i)(\phi_i=\psi_i)$，则称相应的 FEM 为 **Galerkin 型**，否则称其为 **Petrov–Galerkin 型**。
- 若为每个单元独立地选择基函数，并且不要求在单元边界上保证连续性，则称相应的 FEM 为**间断的 (discontinuous)**。
- 计算流体力学中较为常用的是**间断的 Galerkin 型有限单元法 (DG-FEM)**。

为避免频繁对 $\Mat{A}$ 求逆（解线性方程组），可对所选的基函数作**正交化 (orthogonalization)**，使 $\Mat{A}$ 为对角阵。

