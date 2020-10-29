---
title: 电磁学
---

# 实验事实及唯象理论

## 电荷的相互作用力

### Coulomb 扭秤实验 (1785)

### Coulomb 定律<a href name="coulomb"></a>

点电荷 $(q_2,\Vec{r}_2)$ 对点电荷 $(q_1,\Vec{r}_1)$ 的作用力为

$$
\Vec{F}_{2\to1}=\frac{q_1 q_2}{4\pi\epsilon_0}\frac{\Vec{r}_1 - \Vec{r}_2}{\abs{\Vec{r}_1-\Vec{r}_2}^3}
$$

其中 $\epsilon_0$ 为“真空介电常数”。

## 持续的电流

### Volta 电池 (1800)

### Ohm 定律 (1826)

$$
I=U\mathbin{/}R
$$

### Joule--Lenz 定律 (1841--1842)

$$
P = I^2 R
$$



## 电动生磁

### Oersted 电磁感应实验 (1819)

### Biot--Savart 定律 (1820)

电流元 $(I_2\dd{\Vec{l}_2}, \Vec{r}_2)$ 在 $\Vec{r}_1$ 处所诱导出的磁场的磁感应强度为

$$
\dd{\Vec{B}}\mathclose{}(\Vec{r}_1) = \frac{\mu_0 I_2}{4\pi}\dd{\Vec{l}_2}\cross\frac{\Vec{r}_1 - \Vec{r}_2}{\abs{\Vec{r}_1-\Vec{r}_2}^3}
$$

其中 $\mu_0$ 为“真空磁导率”。

## 磁场对运动电荷的作用力

### Ampere 定律 (1820)

磁场 $\Vec{B}$ 对电流元 $I\dd{\Vec{l}}$ 的作用力为

$$
\dd{\Vec{F}} = I \dd{\Vec{l}} \cross \Vec{B}
$$

### Lorentz 力

磁场 $\Vec{B}$ 对运动电荷 $(q,\Vec{v})$ 的作用力为

$$
\Vec{F} = q\Vec{v}\cross\Vec{B}
$$


## 磁动生电

### Faraday 定律 (1831)

$$
\mathcal{E}=-\dv{\varPhi_B}{t}
$$

其中
- $\mathcal{E}\coloneqq\oint_{C} \Vec{E}\vdot\dd{\Vec{l}}$ 为感生电场 $\Vec{E}$ 沿回路 $C$ 形成的感生电动势。
- $\varPhi_B\coloneqq\int_S\Vec{B}\vdot\dd{\Vec{S}}$ 为穿过以 $C$ 为边界的任意曲面 $S$ 的磁通量。

# Maxwell 方程组

## 静止电荷产生的电场

### 点电荷

[Coulomb 定律](#coulomb)可以被改写为
$$
\Vec{F}_{2\to 1}=q_1 \Vec{E}_{2\to 1}
$$

其中

$$
\Vec{E}_{2\to 1}
\equiv\Vec{E}(\Vec{r}_1;q_2,\Vec{r}_2)
\coloneqq\frac{q_2}{4\pi\varepsilon_0}\frac{\Vec{r}_1-\Vec{r}_2}{\vert\Vec{r}_1-\Vec{r}_2\vert^3}
$$

被定义为“点电荷 $q_2\delta(\Vec{r}-\Vec{r}_2)$ 在 $\Vec{r}_1$ 处产生的电场强度”。

实验结果显示“电场强度具有可加性”，即

$$
\Vec{E}(\Vec{r};q_1,\Vec{r}_1\dots,q_n,\Vec{r}_n)
=\sum_{i=1}^n\Vec{E}(\Vec{r};q_i,\Vec{r}_i)
=\sum_{i=1}^n\frac{q_i}{4\pi\varepsilon_0}\frac{\Vec{r}-\Vec{r}_i}{\vert\Vec{r}-\Vec{r}_i\vert^3}
$$

### 一般电荷分布

上述求和式可以被推广为积分式，从而得到“在区域 $V$ 中按 $\rho(\Vec{r}')$ 分布的电荷在 $\Vec{r}$ 处产生的电场强度”：

$$
\Vec{E}(\Vec{r};\rho,V)
=\int_{V(\Vec{r}')}\frac{\rho(\Vec{r}')}{4\pi\varepsilon_0}\frac{\Vec{r}-\Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^3}
$$

### Gauss 电通量定律

按 $\rho(\Vec{r})$ 分布的电荷所产生的电场 $\Vec{E}(\Vec{r})$ 对任意闭合曲面 $S$ 的通量，等于 $S$ 所围区域 $V$ 内的电荷量 $Q$ 除以 $\varepsilon_0$，此即“积分形式的 Gauss 电通量定律”：

$$
\oint_{S=\partial V} \Vec{\nu}\vdot\Vec{E}
= \int_V\frac{\rho}{\varepsilon_0}
= \frac{Q}{\varepsilon_0}
$$

在圆球上应用该式及 Gauss 散度定理，并令圆球半径趋于零，即得其“微分形式”：

$$
\divg\Vec{E}=\frac{\rho}{\varepsilon_0}
$$

此即真空中的 Maxwell 方程组的第一式。

### 静电场无旋有势

静电场 $\Vec{E}(\Vec{r})$ 沿任意闭合曲线 $C$ 的环量为零，即

$$
\oint_C \Vec{\tau}\vdot\Vec{E} = 0
$$

在圆环上应用该式及 Stokes 环量公式，并令圆环半径趋于零，即得

$$
\curl\Vec{E}=\Vec{0}
$$

此即真空中的 Maxwell 方程组的第二式。

静电场无旋的一个推论是：存在标量场 $\phi(\Vec{r})$ 使得 $\Vec{E}(\Vec{r})=-\grad\phi(\Vec{r})$ 对 $\Vec{r}$ 都成立。
这样的标量场 $\phi(\Vec{r})$ 被定义为静电场 $\Vec{E}(\Vec{r})$ 的“静电势”，其中负号是根据 George Green 的建议引入的。
于是，Gauss 电通量定律可以被改写为
$$
\divg\grad\phi\equiv\nabla^2\phi=-\frac{\rho}{\varepsilon_0}
$$

它被称为“Poisson 方程”，其中 $\nabla^2$ 为“Laplace 算符” ，在直角坐标系和球坐标系下的表达式分别为

$$
\nabla^2
=\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}
+\frac{\partial^2}{\partial z^2}
=\frac{1}{r^2}\pdv{}{r}\left(r^2\pdv{}{r}\right)
+\frac{1}{r^2\sin\theta}\pdv{}{r}\left(\sin\theta\pdv{}{\theta}\right)
+\frac{1}{r^2\sin^2\theta}\frac{\partial^2}{\partial\phi^2}
$$

例如：一般电荷分布 $(\rho,V)$ 产生的静电势为

$$
\phi(\Vec{r})
= \frac{1}{4\pi\varepsilon_0}\int_{V(\Vec{r}')}\frac{\rho(\Vec{r}')}{\vert\Vec{r}-\Vec{r}'\vert}
$$

# 狭义相对论

# 静电磁场

# 电磁波