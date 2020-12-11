---
title: 量子力学
---

# 经典物理的量子化

## 黑体辐射

“黑体 (black body)”是指将射在其上的电磁波完全吸收的物体（如：空腔）。温度为 $T$ 的黑体，会以电磁波的形式向外释放能量（如：空腔内壁向空腔内部放出电磁辐射），用 $E_\nu \dd{\nu}$ 表示该温度下、单位体积内、频率介于 $(\nu,\nu+\dd{\nu})$ 之间的电磁波的能量。

### Wien's 公式

Wien (1896) 从热力学理论出发，并结合实验数据，给出了在高频段与实验符合较好、在低频段与实验偏离较大的半经验公式：

$$
E_\nu=C_1\nu^3\exp(-C_2\nu/T)
$$

### Jeans–Rayleigh's 公式

Jeans--Rayleigh 基于电磁驻波理论，给出了在低频段与实验符合较好、在高频段与实验偏离较大的公式：

$$
E_\nu=(8\pi/c^3)(kT)\nu^2\sim T\nu^2
$$

### Planck's 公式

Max Planck (1900) 基于 Wien's 公式“猜”出以下（正确的）公式：

$$
\boxed{E_{\nu}=\frac{C_{2}k\nu}{\exp(C_{2}k\nu/kT)-1}\frac{C_{1}\nu^{2}}{C_{2}k}}
$$

其中

$$
\bar{\varepsilon}_{\nu}\coloneqq\frac{C_{2}k\nu}{\exp(C_{2}k\nu/kT)-1}\qquad N_{\nu}\coloneqq\frac{C_{1}\nu^{2}}{C_{2}k}
$$

分别表示频率介于 $(\nu,\nu+\dd{\nu})$ 之间的“允许模式”的“平均能量”与“个数”。

## 光电效应

### Hertz's 电磁波速实验

### Hertz's 光电效应实验

### Einstein's 光量子

Albert Einstein (1905) 提出：频率为 $\nu$ 的光由大量能量为 $h\nu$ 的“光量子 (quanta of light)”组成。由“能量守恒”及“动能非负”

$$
\boxed{h\nu-A=\frac{1}{2}mv^{2}\ge0}
$$

可得“临界频率”

$$
\nu_{0}\coloneqq A/h\le\nu
$$

基于狭义相对论，Einstein 又 (1915) 提出：光量子的静止质量为零，从而有

$$
\vert\Vec{p}\vert=p=\frac{E}{c}=\frac{h\nu}{c}=\frac{h}{2\mathrm{\pi}}\frac{2\mathrm{\pi}}{\lambda}=\hbar k=\hbar\vert\Vec{k}\vert
$$

后世称这种粒子为“光子 (photon)”。

### Compton's 散射实验

## 原子线状光谱

### 光栅

### 光谱

### Balmer's 光谱公式

$$
\boxed{\frac{1}{\lambda_{2,n}}=\left(\frac{1}{2^{2}}-\frac{1}{n^{2}}\right)R\qquad n=3,4,\dots}
$$

## 原子稳定性

### Thomson's 原子模型

### Rutherford's 散射实验

## Bohr's 轨道论

### 基本假设

- 电子沿一系列分立的椭圆轨道绕原子核运动。每一条这样的轨道对应于该系统的一个“定态”。
- 氢原子状态在两个定态之间发生“跃迁”时，吸收或放出的电磁波满足 Planck–Einstein 条件：$ h\nu_{m\to n} = E_m - E_n $
- 当量子数 $ n \to \infty $ 时，应当接近经典物理给出的数值。该假设被称为“对应原理”。

### 椭圆轨道

$$
E=-\frac{\kappa}{2a}<0\qquad\frac{T^{2}}{a^{3}}=\frac{4\mathrm{\pi}^{2}m}{\kappa}
$$

$$
\nu=\frac{1}{T}=\frac{\sqrt{2\vert E\vert^{3}/m}}{\mathrm{\pi}\kappa}
$$

### 氢原子能级

$$
\boxed{E_{n}=-\frac{2\mathrm{\pi}^{2}e^{4}m}{h^{2}n^{2}}\qquad n=1,2,3,\dots}
$$

$$
h\nu_{n\to2}=\left(\frac{1}{2^{2}}-\frac{1}{n^{2}}\right)\frac{2\mathrm{\pi}^{2}e^{4}m}{h^{2}n^{2}}\qquad n=3,4,5,\dots
$$

### 角动量量子化

考虑 $r=R$ 的圆轨道，并引入角动量

$$
\Vec{L}=\Vec{r}\times m\Vec{v}\implies L=R\cdot mR\dv{\theta}{t}=mR^{2}\dv{\theta}{t}
$$

代入

$$
E_n=\frac{mR^{2}}{2}\left(\dv{\theta}{t}\right)^{2}-\frac{e^{2}}{R}=\frac{L^{2}}{2mR^{2}}-\frac{e^{2}}{R}=\frac{e^2}{2R}
$$

即得

$$
\boxed{L=\frac{nh}{2\mathrm{\pi}}=n\hbar\qquad n=1,2,3,\dots}
$$

Sommerfeld 量子化条件：

$$
\boxed{\oint p\dd{q}=n\hbar\qquad n=1,2,3,\dots}
$$

## Heisenburg's 矩阵力学

### 一人文章

Werner Heisenburg (1925) 指出：物理理论应该在建立在与实验观测紧密关联的物理量上。所有可观测的物理量都与两条（而非一条）Bohr 轨道关联。

以氢原子光谱为例：氢原子从第 $m$ 态到第 $n$ 态跃迁所放出的电磁波的频率为  $\nu_{mn} $，所有这样的（由两个状态决定的）频率可写成如下矩阵

$$
\hat{\nu}=\begin{bmatrix}\nu_{11} & \cdots & \nu_{1n} & \cdots\\
\vdots & \ddots & \vdots\\
\nu_{m1} & \cdots & \nu_{mn} & \cdots\\
\vdots &  & \vdots & \ddots
\end{bmatrix}
$$

### 二人文章

Pascaul Jordan 与 Max Born 利用一维谐振子的特性，得到

$$
[\hat{x},\hat{p}]\equiv\hat{x}\hat{p}-\hat{p}\hat{x}=\sqrt{-1}\hbar
$$

### 三人文章

## Dirac's 正则量子化方法

### 对易恒等式

Paul Dirac 回忆起，分析力学中的 Poisson's 括号

$$
\{u,v\}\coloneqq\sum_{i=1}^{n}\left(\frac{\partial u}{\partial q_{i}}\frac{\partial v}{\partial p_{i}}-\frac{\partial u}{\partial p_{i}}\frac{\partial v}{\partial q_{i}}\right)
$$

亦满足类似的恒等式。根据这种相似性，Dirac 提出了如下“Dirac's 方程”

$$
[\hat{u},\hat{v}]=\{u,v\}\hat{D}
$$

量子力学基本关系式：

$$
[\hat{q}_{i},\hat{q}_{k}]=\hat{0}\qquad[\hat{p}_{i},\hat{p}_{k}]=\hat{0}\qquad[\hat{q}_{i},\hat{p}_{k}]=\mathrm{\delta}_{ik}\hat{D}
$$

### Pauli 恒等式

### 氢原子光谱

$$
E(n)=\frac{me^{4}}{2n^{2}D^{2}}=\frac{2\mathrm{\pi}^{2}me^{4}}{-n^{2}h^{2}}\implies\boxed{D=\sqrt{\frac{-h^{2}}{4\mathrm{\pi}^{2}}}=\sqrt{-1}\hbar}
$$

### 正则量子化

# 用波函数描述量子态

# 用算符描述可观测量

# 一维势场中的粒子

# 中心力场中的粒子