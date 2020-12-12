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
[\widehat{q_{i}},\widehat{q_{k}}]=\hat{0}\qquad[\widehat{p_{i}},\widehat{p_{k}}]=\hat{0}\qquad[\widehat{q_{i}},\widehat{p_{k}}]=\mathrm{\delta}_{ik}\hat{D}
$$

### Pauli 恒等式

### 氢原子光谱

$$
E(n)=\frac{me^{4}}{2n^{2}D^{2}}=\frac{2\mathrm{\pi}^{2}me^{4}}{-n^{2}h^{2}}\implies\boxed{D=\sqrt{\frac{-h^{2}}{4\mathrm{\pi}^{2}}}=\sqrt{-1}\hbar}
$$

### 正则量子化

- 写出系统的 Lagranian 函数 $\boxed{L(\underline{q},\underline{\dot{q}})=T(\underline{q},\underline{\dot{q}},t)-V(\underline{q},t)}$，其中 $T,V$ 分别为系统的“动能”与“势能”。
- 定义广义动量 $\boxed{\underline{p}=\partial L/\partial\underline{\dot{q}}}$，并将 $\underline{\dot{q}}$ 用 $\underline{q},\underline{p}$ 表示。
- 定义 Hamiltonian 函数 $\boxed{H(\underline{q},\underline{p},t)=\underline{p}\cdot\underline{q}-L(\underline{q},\underline{\dot{q}},t)}$，其中 $\underline{p}\cdot\underline{q}=\sum_{i=1}^np_iq_i$ 表示逐项相乘。
- 将物理量 $H,\underline{q},\underline{p}$ 替换为相应的算符 $\hat{H},\underline{\hat{q}},\underline{\hat{p}}$，并要求 $[\widehat{p_{i}},\widehat{q_{k}}]=\mathrm{\delta}_{ik}\sqrt{-1}\hbar\hat{1}$，即可写出该系统的 Schrödinger 方程：

$$
\boxed{\sqrt{-1}\hbar\pdv{}{t}\psi(\underline{q},\underline{p},t)=\hat{H}(\underline{\hat{q}},\underline{\hat{p}},t)\,\psi(\underline{q},\underline{p},t)}
$$

# 用波函数描述量子态

## 波粒二象性

### Planck's 能量子

### Einstein's 光量子

### De Broglie's 物质波

基于光子概念，Louis de Broglie (1924) 在其博士论文中（大胆地）提出了“一般粒子也具有波粒二象性”的假设：

- 单个自由粒子的状态可以由自变量为位置 $\Vec{r}$ 和时间 $ t $ 的复值函数 $\psi(\Vec{r},t)$ 来描述。
- 能量为 $E$、动量为 $\Vec{p}$ 的单个“自由粒子”的状态可以由“平面波” $ \boxed{\psi(\Vec{r},t)=A\exp(\mathrm{i}\omega t-\mathrm{i}\Vec{k}\vdot\Vec{r})} $ 来描述，其中 $A$ 为复常数。
- 描述粒子性的能量 $E$、动量 $\Vec{p}$ 与描述波动性的圆频率 $\omega$、波矢 $\Vec{k}$ 之间，有类似于光子的“波粒二象性”关系 $
\boxed{(E,\Vec{p})=(\hbar\omega,\hbar\Vec{k})} $，其中 $\hbar\coloneqq h/(2\mathrm{\pi})$ 为“约化的 Planck 常量”。

上述假设后来被“电子双缝干涉”等实验所证实。

## 几率振幅

### Schrödinger 方程

Erwin Schrödinger (1927) 从 de Broglie's 平面波这一特例出发，导出了如下“Schrödinger 方程”：

$$
\mathrm{i}\hbar\pdv{}{t}\psi(\Vec{r},t)=\left(\frac{-\hbar^{2}}{2m}\nabla^{2}+V(\Vec{r})\right)\psi(\Vec{r},t)
$$

满足此方程的波函数 $\psi(\Vec{r},t)$ 描述了单个粒子的一种“许可状态”。

### 统计解释

波函数的物理意义至今仍无定论。目前最主流的观点是 Max Born (1926) 提出的“统计解释 (statistical interpretation)”：

- 某粒子的波函数 $\psi$ 又被称为其“几率振幅 (probability amplitude)”，它在 $(\Vec{r},t)$ 处的模平方，即 $\vert\psi(\Vec{r},t)\vert^{2}$，正比于 $t$ 时刻在 $\Vec{r}$ 处发现该粒子的几率密度。
- 只要 $\psi$ 所描述的那个粒子不发生湮灭，则“该粒子位于全空间”是一个“必然事件”，即 $\int_{\mathbb{R}^{3}(\Vec{r})}\vert\psi(\Vec{r},t)\vert^{2}=1$，该式称为波函数的“归一化条件”。
- 然而，并不是所有波函数都可以被归一化（例如平面波），此时 $\vert\psi(\Vec{r},t)\vert^{2}$ 应理解为相对几率密度。

### 几率守恒

几率密度满足“连续性方程”，即

$$
\frac{\partial\rho}{\partial t}+\divg\Vec{\jmath}=0
$$

其中

$$
\rho\coloneqq\psi^{*}\psi\equiv\vert\psi\vert^{2}\qquad\Vec{\jmath}\coloneqq\frac{-\mathrm{i}\hbar}{2m}(\psi^{*}\grad\psi-\psi\grad\psi^{*})
$$

### 态叠加原理

【公理】若波函数 $\psi_{1},\psi_{2}$ 描述了某量子系统的两种许可态，则波函数 $C_{1}\psi_{1}+C_{2}\psi_{2}$ 亦为该系统的一种许可态，其中 $C_{1},C_{2}$ 为复常数。

【干涉】一般而言，几率密度 $ \psi^{*}\psi $ 不满足叠加原理，而是遵循以下法则：

$$
\begin{aligned}\vert C_{1}\psi_{1}+C_{2}\psi_{2}\vert^{2} & =(C_{1}\psi_{1}+C_{2}\psi_{2})^{*}\,(C_{1}\psi_{1}+C_{2}\psi_{2})\\
 & =\underbrace{(C_{1}\psi_{1})^{*}\,(C_{1}\psi_{1})}_{\rho_{1}}+\underbrace{(C_{2}\psi_{2})^{*}\,(C_{2}\psi_{2})}_{\rho_{2}}+\underbrace{(C_{1}\psi_{1})^{*}\,(C_{2}\psi_{2})+(C_{2}\psi_{2})^{*}\,(C_{1}\psi_{1})}_{\rho_{1,2}}
\end{aligned}
$$

其中 $\rho_{1,2}$ 就体现了波的“干涉 (interference)”。

## 态矢量

### Hilbert 空间

根据态叠加原理，一个量子系统的所有许可态构成了一个 $\mathbb{C}$ 上的线性空间，用 $X$ 表示。

【向量】$X$ 中的每一个状态 $\psi(\Vec{r},t)$ 都是一个向量，记作 $\vert\psi\rangle$，称为“态矢量 (state vector)”。

【基底】若存在 $X$ 中的一组线性独立的状态 $\begin{Bmatrix} \vert\psi_\alpha\rangle \end{Bmatrix}_{\alpha\in A}$，使得 $X$ 中的任何一个矢量 $\vert\psi\rangle$ 都可以表示成它们的线性组合，即

$$
\vert\psi\rangle=\begin{cases}
\sum_{\alpha\in A}C_{\alpha}\mathinner{\vert\psi_{\alpha}\rangle} & A\sim\mathbb{N}\\
\int_{A}C_{\alpha}\mathinner{\vert\psi_{\alpha}\rangle}\dd{\alpha} & A\sim\mathbb{R}
\end{cases}
$$

则称 $\begin{Bmatrix} \vert\psi_\alpha\rangle \end{Bmatrix}_{\alpha\in A}$ 为 $X$ 的一组“基底 (basis)”，其中每一个 $\vert\psi_\alpha\rangle$ 称为一个“基矢量 (basis vector)”。

【维数】一般而言，指标集 $A$ 含有无穷多个成员，因此 $X$ 通常是无穷维的。

【内积】
$$
\langle u\vert v\rangle\coloneqq\int_{-\infty}^{+\infty}u^{*}(x)\,v(x)\dd{x}
$$

【平方可积】
$$
\langle \psi\vert \psi\rangle\coloneqq\int_{-\infty}^{+\infty}\psi^{*}(x)\,\psi(x)\dd{x}<\infty
$$



### 坐标表象

一维驻波的波函数 $\psi(x)$ 总是可以展开为 $\begin{Bmatrix} \psi_\alpha(x)\coloneqq\mathrm{\delta}(x-\alpha) \end{Bmatrix}_{\alpha\in\mathbb{R}}$ 的线性组合：

$$
\psi(x)=\int_{-\infty}^{+\infty}\psi(\alpha)\,\mathrm{\delta}(x-\alpha)\dd{\alpha}=\int_{-\infty}^{+\infty}C_{\alpha}\,\psi_\alpha(x)\dd{\alpha}
$$

指标 $\alpha$ 与函数 $\psi_\alpha$ 一一对应，故可引入简化记号

$$
\vert\alpha\rangle\coloneqq\vert\psi_\alpha\rangle\qquad\langle\alpha\vert\coloneqq\langle\psi_\alpha\vert
$$

从而有

$$
\boxed{\psi(\alpha)=\langle\psi_\alpha\vert\psi\rangle=\langle\alpha\vert\psi\rangle}\qquad\boxed{\mathrm{\delta}(x-\alpha)=\psi_\alpha(x)=\langle x\vert\psi_\alpha\rangle=\langle x\vert\alpha\rangle}
$$

于是，可将波函数的展开式改写为

$$
\langle x\vert\psi\rangle=\int_{-\infty}^{+\infty}\langle\alpha\vert\psi\rangle\langle x\vert\alpha\rangle\dd{\alpha}=\langle x\vert\left(\int_{-\infty}^{+\infty}\dyad{\alpha}\dd{\alpha}\right)\vert\psi\rangle
$$

比较两端可发现“完备性条件 (completeness condition)”：

$$
\boxed{\hat{1}=\int_{-\infty}^{+\infty}\dyad{\alpha}\dd{\alpha}}
$$

### 动量表象

一维驻波 $\psi(x)$ 除了可以按坐标分解，还可以按波数分解为无穷多个简谐驻波的线性组合：

$$
\psi(x)\doteq\int_{-\infty}^{+\infty}\tilde{\psi}(k)\frac{\exp(+\mathrm{i} kx)}{\sqrt{2\mathrm{\pi}}}\dd{k}\qquad\tilde{\psi}(k)\coloneqq\int_{-\infty}^{+\infty}\psi(x)\frac{\exp(-\mathrm{i} kx)}{\sqrt{2\mathrm{\pi}}}\dd{x}
$$

在数学中，此二式被称为“Fourier 变换对”。

坐标空间 $\mathbb{R}(x)$ 上的函数 $\psi(x)$ 与波数空间 $\mathbb{R}(k)$ 上的函数 $\tilde{\psi}(k)$ 可以被看作同一个对象 $\vert\psi\rangle$ 的两种几乎等价的“表象 (representation)”。

在量子力学中，更常用的是“动量表象”，它可以通过将 de Broglie 关系 $\Vec{p}=\hbar\Vec{k}$ 代入“Fourier 变换对”得到：

$$
\boxed{\begin{gathered}\tilde{\psi}(\Vec{p})=\langle\Vec{p}\vert\psi\rangle=\int_{\mathbb{R}^{n}(\Vec{r})}\langle\Vec{p}\vert\Vec{r}\rangle\langle\Vec{r}\vert\psi\rangle\qquad\psi(\Vec{r})=\langle\Vec{r}\vert\psi\rangle=\int_{\mathbb{R}^{n}(\Vec{p})}\langle\Vec{r}\vert\Vec{p}\rangle\langle\Vec{p}\vert\psi\rangle\\
\langle\Vec{p}\vert\Vec{r}\rangle\coloneqq\frac{\exp(-(\mathrm{i}/\hbar)\,\Vec{p}\vdot\Vec{r})}{(2\mathrm{\pi}\hbar)^{n/2}}\qquad\langle\Vec{r}\vert\Vec{p}\rangle\coloneqq\frac{\exp(+(\mathrm{i}/\hbar)\,\Vec{p}\vdot\Vec{r})}{(2\mathrm{\pi}\hbar)^{n/2}}
\end{gathered}
}
$$

这里为了保证 $\Vec{r}$ 与 $\Vec{p}$ 的对称性 $\langle\Vec{p}\vert\Vec{r}\rangle=\langle\Vec{r}\vert\Vec{p}\rangle^{*}$，在 $\langle\Vec{p}\vert\Vec{r}\rangle$ 与 $\langle\Vec{r}\vert\Vec{p}\rangle$ 之间重新分配了系数。

# 用算符描述可观测量

# 一维势场中的粒子

# 中心力场中的粒子