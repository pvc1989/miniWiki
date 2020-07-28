---
title: 流体力学
---

# 流动方程

## 速度导数

### 速度梯度

$$
\dd{\Vec{u}}=\dd{\Vec{x}}\vdot\grad\Vec{u}
$$

$$
\grad\Vec{u}=\left(\Vec{e}_{i}\pdv{}{x_{i}}\right)\otimes\left(\Vec{e}_{k}u_{k}\right)=\left(\Vec{e}_{i}\otimes\Vec{e}_{k}\right)\pdv{u_{k}}{x_{i}}=\Vec{e}_{i}\Vec{e}_{k}\partial_{i}u_{k}
$$

$$
\boxed{\grad\Vec{u}=\VecVec{E}+\VecVec{\varOmega}}\impliedby\begin{cases}
\VecVec{E}\coloneqq\Vec{e}_{i}\Vec{e}_{k}E_{ik} & E_{ik}\coloneqq(\partial_{i}u_{k}+\partial_{k}u_{i})/2\\
\VecVec{\varOmega}\coloneqq\Vec{e}_{i}\Vec{e}_{k}\varOmega_{ik} & \varOmega_{ik}\coloneqq(\partial_{i}u_{k}-\partial_{k}u_{i})/2
\end{cases}
$$

### 角速度

$$
\VecVec{\varOmega}=\begin{bmatrix}\Vec{e}_{1} & \Vec{e}_{2} & \Vec{e}_{3}\end{bmatrix}\begin{bmatrix}0 & +\varOmega_{12} & -\varOmega_{31}\\
-\varOmega_{12} & 0 & +\varOmega_{23}\\
+\varOmega_{31} & -\varOmega_{23} & 0
\end{bmatrix}\begin{bmatrix}\Vec{e}_{1}\\
\Vec{e}_{2}\\
\Vec{e}_{3}
\end{bmatrix}
$$

$$
\begin{bmatrix}\varOmega_{1}\\
\varOmega_{2}\\
\varOmega_{3}
\end{bmatrix}=\begin{bmatrix}\varOmega_{23}\\
\varOmega_{31}\\
\varOmega_{12}
\end{bmatrix}\iff\varOmega_{jk}=\varOmega_{i}\epsilon_{ijk}\quad\text{in which}\quad\epsilon_{ijk}=\begin{cases}
+1 & ijk=123,231,312\\
-1 & ijk=321,132,213\\
0  & \text{else}
\end{cases}
$$

$$
\boxed{\dd{\Vec{x}}\vdot\VecVec{\varOmega}=\Vec{\varOmega}\cross\dd{\Vec{x}}}\impliedby\begin{cases}
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
\boxed{\dv{}{t}\int_{V}\phi=\int_{V}\pdv{\phi}{t}+\oint_{\partial V}\Vec{n}\vdot\Vec{u}_{\mathrm{CS}}\,\phi}
$$

### 控制体上的物质导数

$$
\boxed{\begin{aligned}\dv{}{t}\int_{V_{\mathrm{MB}}}\phi\eqqcolon\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\phi & =\int_{V}\pdv{\phi}{t}+\oint_{\partial V}\Vec{n}\vdot\Vec{u}\phi\\
 & =\dv{}{t}\int_{V}\phi+\oint_{\partial V}\Vec{n}\vdot\left(\Vec{u}-\Vec{u}_{\mathrm{CS}}\right)\phi
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

|          |                            积分型                            |                微分型                |
| :------: | :----------------------------------------------------------: | :----------------------------------: |
|  守恒型  | $\int_{V}\pdv{\rho}{t}+\oint_{\partial V}\Vec{n}\vdot\Vec{u}\rho=0$ | $\pdv{\rho}{t}+\divg(\Vec{u}\rho)=0$ |
| 非守恒型 | $\dv{}{t}\int_{V}\rho+\oint_{\partial V}\Vec{n}\vdot\left(\Vec{u}-\Vec{u}_{\mathrm{CS}}\right)\rho=0$ |   $\frac{\mathrm{D}\rho}{\mathrm{D}t}+\rho\divg\Vec{u}=0$    |

### 动量守恒

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho\Vec{u}=\int_{V}\Vec{b}+\oint_{\partial V}\Vec{n}\vdot\VecVec{\sigma}}
$$

其中 $\VecVec{\sigma}$ 为 ***应力张量 (stress tensor)***，它与单位法向量 $\Vec{n}$ 的点乘 $\Vec{n}\vdot\VecVec{\sigma}$ 是一个向量，表示作用在物质体边界 $\partial V_{\mathrm{MB}}$ 上的外力的面密度。在流体力学中，应力张量 $\VecVec{\sigma}$ 总是被分解为 ***静水压力 (hydrostatic pressure) 张量*** $-p\VecVec{1}$ 与 ***黏性应力 (viscous stress) 张量*** $\VecVec{\tau}$ 之和：

$$
\VecVec{\sigma}=-p\VecVec{1}+\VecVec{\tau}\iff\sigma_{ik}=-p\delta_{ik}+\tau_{ik}
$$

于是动量守恒定律可以改写为

$$
\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho\Vec{u}+\oint_{\partial V}\Vec{n}p=\int_{V}\Vec{b}+\oint_{\partial V}\Vec{n}\vdot\VecVec{\tau}
$$

### 角动量守恒

在（不考虑分布式力偶的）流体力学中，它等价于 *应力张量* 或 *黏性应力张量* 的对称性：

$$
\boxed{\underbrace{-p\delta_{ik}+\tau_{ik}}_{\sigma_{ik}}=\underbrace{-p\delta_{ki}+\tau_{ki}}_{\sigma_{ki}}}
$$

### 能量守恒

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\rho e_{0}=\int_{V}\left(\Vec{b}\vdot\Vec{u}+j\right)+\oint_{\partial V}\Vec{n}\vdot\left(\VecVec{\sigma}\vdot\Vec{u}-\Vec{q}\right)}
$$

其中
- 标量 $e_0\coloneqq e+\abs{\Vec{u}}^{2}/2$ 为 ***比总能 (specific total energy)***，表示 *单位质量* 流体的 *内能* 与 *动能* 之和。
- 标量 $j$ 为 ***热源密度 (density of heat source)***，表示 *单位时间* 内 *单位体积* 热源的 ***热生成量 (heat generation)***。
- 向量 $\Vec{q}$ 为 ***热通量密度 (density of heat flux)***，表示 *单位时间* 内穿过 *单位面积* 的 ***热流量 (heat flow)***。

## Navier–Stokes 方程组

将守恒定律整理成矩阵（方程组）形式，即得 *Navier–Stokes 方程组*

$$
\boxed{\frac{\mathrm{D}}{\mathrm{D}t}\int_{V}\ket{U}+\oint_{\partial V}\Vec{n}\vdot\ket{\Vec{P}}=\oint_{\partial V}\Vec{n}\vdot\ket{\Vec{G}}+\int_{V}\ket{H}}
$$

其中

$$
\ket{U}=\begin{bmatrix}\rho\\
\rho\Vec{u}\\
\rho e_{0}
\end{bmatrix}\qquad\ket{\Vec{P}}=\begin{bmatrix}\Vec{0}\\
p\VecVec{1}\\
p\Vec{u}
\end{bmatrix}\qquad\ket{\Vec{G}}=\begin{bmatrix}\Vec{0}\\
\VecVec{\tau}\\
\VecVec{\tau}\vdot\Vec{u}-\Vec{q}
\end{bmatrix}\qquad\ket{H}=\begin{bmatrix}0\\
\Vec{b}\\
\Vec{b}\vdot\Vec{u}+j
\end{bmatrix}
$$

### 守恒型

利用[控制体上的物质导数](#控制体上的物质导数)的展开式，并引入 ***通量 (flux) 矩阵***

$$
\ket{\Vec{F}}\coloneqq\ket{U}\Vec{u}+\ket{\Vec{P}}=\begin{bmatrix}\rho\\
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

即得 *守恒型积分方程组*

$$
\int_{V}\partial_{t}\ket{U}+\oint_{\partial V}\Vec{n}\vdot\ket{\Vec{F}}=\oint_{\partial V}\Vec{n}\vdot\ket{\Vec{G}}+\int_{V}\ket{H}
$$

对其中的面积分应用 Gauss 散度定理，即得 *守恒型微分方程组*

$$
\partial_{t}\ket{U}+\divg\ket{\Vec{F}}=\divg\ket{\Vec{G}}+\ket{H}
$$

它在三维直角坐标系中的分量形式为

$$
\pdv{}{t}\begin{bmatrix}\rho\\
\rho u_{1}\\
\rho u_{2}\\
\rho u_{3}\\
\rho e_{0}
\end{bmatrix}+\pdv{}{x_{\alpha}}\begin{bmatrix}\rho u_{\alpha}\\
\rho u_{1}u_{\alpha}+p\delta_{1\alpha}\\
\rho u_{2}u_{\alpha}+p\delta_{2\alpha}\\
\rho u_{3}u_{\alpha}+p\delta_{3\alpha}\\
\rho h_{0}u_{\alpha}
\end{bmatrix}=\pdv{}{x_{\alpha}}\begin{bmatrix}0\\
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

注意到守恒项 $\ket{U}$ 中的 $\rho$ 可以被提出：

$$
\ket{U}=\rho\ket{W} \impliedby \ket{W}\coloneqq\begin{bmatrix}1 \\ \vec{u} \\ e_{0}\end{bmatrix}
$$

故[守恒型积分方程组](#守恒型)中的[物质导数](#物质导数)可移入积分号内，由此即得 *非守恒型积分方程组*

$$
\int_{V}\rho\,\overbrace{\left(\partial_{t}+\Vec{u}\vdot\grad\right)}^{\mathrm{D}_t}\ket{W}+\oint_{\partial V}\vec{n}\vdot\ket{\Vec{P}}=\oint_{\partial V}\vec{n}\vdot\ket{\Vec{G}}+\int_{V}\ket{H}
$$

相应地有 *非守恒型微分方程组*

$$
\rho\left(\partial_{t}+\Vec{u}\vdot\grad\right)\ket{W}+\divg\ket{\Vec{P}}=\divg\ket{\Vec{G}}+\ket{H}
$$

它在三维直角坐标系中的分量形式为

$$
\rho\left(\partial_{t}+\Vec{u}\vdot\grad\right)
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
\end{bmatrix}=\partial_{\alpha}\begin{bmatrix}0\\
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

$$
\boxed{\pdv{U}{t}+\divg\Vec{F}=0}
$$

$$
\int_{\varOmega}\left(\pdv{U}{t}+\divg\Vec{F}\right)V=0\qquad\forall V,\forall\varOmega
$$

$$
\int_{\varOmega}V\pdv{U}{t}
=\int_{\varOmega}\Vec{F}\vdot\grad V-\oint_{\partial\varOmega}\left(\Vec{\nu}\vdot\Vec{F}\right)V\qquad\forall V,\forall\varOmega
$$

### 正交基函数

$$
U(\Vec{x},t)\approx U^{h}(\Vec{x},t)=\sum_{k=1}^{K}\hat{U}_{k}(t)\,\phi_{k}(\Vec{x})\qquad V(\Vec{x})\approx V^{h}(\Vec{x})=\sum_{l=1}^{L}\hat{V}_{l}\,\psi_{l}(\Vec{x})
$$

$$
\underbrace{\begin{bmatrix}\ip{\psi_{1}}{\phi_{1}} & \cdots & \ip{\psi_{1}}{\phi_{K}}\\
\vdots & \ddots & \vdots\\
\ip{\psi_{L}}{\phi_{1}} & \cdots & \ip{\psi_{L}}{\phi_{K}}
\end{bmatrix}}_{\Mat{A}}\dv{}{t}\underbrace{\begin{bmatrix}\hat{U}_{1}\\
\vdots\\
\hat{U}_{K}
\end{bmatrix}}_{\ket{U}}=\underbrace{\int_{\varOmega}\begin{bmatrix}\grad\psi_{1}\\
\vdots\\
\grad\psi_{L}
\end{bmatrix}\vdot\Vec{F}\mathopen{}\left(U^{h}\right)-\oint_{\partial\varOmega}\begin{bmatrix}\psi_{1}\\
\vdots\\
\psi_{L}
\end{bmatrix}F_{\nu}\mathopen{}\left(U_{-}^{h},U_{+}^{h}\right)}_{\ket{B(U)}}
$$

$$
\boxed{\Mat{A}\dv{}{t}\ket{U}=\ket{B(U)}}
$$

若 $(K=L)$ 且 $(\forall i)(\phi_i=\psi_i)$，则导出的 FEM 为 ***Galerkin 型***，否则为 ***Petrov--Galerkin 型***。

