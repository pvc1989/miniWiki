---
title: 电磁学、电动力学
---

# 实验事实及唯象理论

## 电荷的相互作用力

### Coulomb 扭秤实验 (1785)

### Coulomb 定律<a href name="coulomb"></a>

点电荷 $(q_2,\Vec{r}_2)$ 对点电荷 $(q_1,\Vec{r}_1)$ 的作用力为

$$
\Vec{F}_{2\to1}=\frac{q_1 q_2}{4\pi\epsilon_0}\frac{\Vec{r}_1 - \Vec{r}_2}{\abs{\Vec{r}_1-\Vec{r}_2}^3}
$$

其中 $\epsilon_0$ 为**真空介电常数**。

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

### Biot--Savart 定律 (1820)<a href name="biot--savart"></a>

电流元 $(I_2\dd{\Vec{l}_2}, \Vec{r}_2)$ 在 $\Vec{r}_1$ 处所诱导出的磁场的磁感应强度为

$$
\dd{\Vec{B}(\Vec{r}_1)} = \frac{\mu_0 I_2}{4\pi}\dd{\Vec{l}_2}\cross\frac{\Vec{r}_1 - \Vec{r}_2}{\abs{\Vec{r}_1-\Vec{r}_2}^3}
$$

其中 $\mu_0$ 为**真空磁导率**。

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

# 真空中的 Maxwell 方程组

## 静止电荷产生的静电场

### 静电场的强度

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

被定义为**点电荷 $q_2\delta(\Vec{r}-\Vec{r}_2)$ 在 $\Vec{r}_1$ 处产生的电场强度**。

实验结果显示电场强度具有**可加性**，即

$$
\Vec{E}(\Vec{r};q_1,\Vec{r}_1\dots,q_n,\Vec{r}_n)
=\sum_{i=1}^n\Vec{E}(\Vec{r};q_i,\Vec{r}_i)
=\sum_{i=1}^n\frac{q_i}{4\pi\varepsilon_0}\frac{\Vec{r}-\Vec{r}_i}{\vert\Vec{r}-\Vec{r}_i\vert^3}
$$

上述求和式可以被推广为积分式，从而得到**在区域 $V$ 中按 $\rho(\Vec{r}')$ 分布的电荷在 $\Vec{r}$ 处产生的电场强度**：

$$
\Vec{E}(\Vec{r};\rho,V)
=\int_{V(\Vec{r}')}\frac{\rho(\Vec{r}')}{4\pi\varepsilon_0}\frac{\Vec{r}-\Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^3}
$$

### 静电场的散度

按 $\rho(\Vec{r})$ 分布的电荷所产生的电场 $\Vec{E}(\Vec{r})$ 对任意闭合曲面 $S$ 的通量，等于 $S$ 所围区域 $V$ 内的电荷量 $Q$ 除以 $\varepsilon_0$，此即**积分形式的 Gauss 电通量定律**：

$$
\oint_{S=\partial V} \Vec{\nu}\vdot\Vec{E}
= \int_V\frac{\rho}{\varepsilon_0}
= \frac{Q}{\varepsilon_0}
$$

在圆球上应用该式及 Gauss 散度定理，并令圆球半径趋于零，即得其**微分形式**：

$$
\boxed{\divg\Vec{E}=\frac{\rho}{\varepsilon_0}}
$$

此即**真空中的稳态 Maxwell 方程组**的第一式。

### 静电场的旋度

静电场 $\Vec{E}(\Vec{r})$ 沿任意闭合曲线 $C$ 的环量为零，即

$$
\oint_C \Vec{\tau}\vdot\Vec{E} = 0
$$

在圆环上应用该式及 Stokes 环量公式，并令圆环半径趋于零，即得

$$
\boxed{\curl\Vec{E}=\Vec{0}}
$$

此即真空中的稳态 Maxwell 方程组的第二式。

### 静电场的标量势

静电场无旋的一个推论是：存在标量场 $\phi(\Vec{r})$ 使得 $\Vec{E}(\Vec{r})=-\grad\phi(\Vec{r})$ 对 $\Vec{r}$ 都成立。
这样的标量场 $\phi(\Vec{r})$ 被定义为静电场 $\Vec{E}(\Vec{r})$ 的**静电势**，其中负号是根据 George Green 的建议引入的。
于是，Gauss 电通量定律可以被改写为

$$
\divg\grad\phi\equiv\nabla^2\phi=-\frac{\rho}{\varepsilon_0}
$$

它被称为 **Poisson 方程**，其中 $\nabla^2$ 为 **Laplace 算符** ，在直角坐标系和球坐标系下的表达式分别为

$$
\nabla^2
=\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}
+\frac{\partial^2}{\partial z^2}
=\frac{1}{r^2}\pdv{}{r}\left(r^2\pdv{}{r}\right)
+\frac{1}{r^2\sin\theta}\pdv{}{\theta}\left(\sin\theta\pdv{}{\theta}\right)
+\frac{1}{r^2\sin^2\theta}\frac{\partial^2}{\partial\phi^2}
$$

例如：一般电荷分布 $(\rho,V)$ 产生的静电势为

$$
\phi(\Vec{r})
= \frac{1}{4\pi\varepsilon_0}\int_{V(\Vec{r}')}\frac{\rho(\Vec{r}')}{\vert\Vec{r}-\Vec{r}'\vert}
$$

## 恒定电流产生的静磁场

### 电荷守恒定律

封闭系统内的电荷总量保持不变，此即**电荷守恒定律**。具体的，给定（不随时间变化的）区域 $V$ 及（可能）随时间变化的电荷密度 $\rho(\Vec{r},t)$ 与电流密度 $\Vec{\jmath}(\Vec{r},t)$，则该定律可写为

$$
\int_V\pdv{\rho(\Vec{r},t)}{t}+\oint_{\partial V}\Vec{\nu}(\Vec{r},t)\vdot\Vec{\jmath}(\Vec{r},t)=0
$$

利用 Gauss 散度定理，并令 $V$ 趋于无穷小区域，即得其**微分形式**： 

$$
\boxed{\pdv{\rho(\Vec{r},t)}{t}+\divg\Vec{\jmath}(\Vec{r},t)=0}
$$

### 电流产生的静磁场

由 [Biot--Savart 定律](#biot--savart)可知：闭合回路 $C$ 中的电流所产生的磁场的磁感应强度为

$$
\Vec{B}(\Vec{r})
= \oint_{C(\Vec{r}')}\frac{\mu_0 I(\Vec{r}')}{4\pi}\dd{\Vec{l}(\Vec{r}')}\cross\frac{\Vec{r} - \Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^3}
$$

以及更一般的

$$
\Vec{B}(\Vec{r})
= \frac{\mu_0}{4\pi}\int_{V(\Vec{r}')}\Vec{\jmath}(\Vec{r}')\cross\frac{\Vec{r} - \Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^3}
$$

### 静磁场的矢量势

利用微分恒等式

$$
\grad\frac{-1}{\vert\Vec{r}-\Vec{r}'\vert}
=\frac{\Vec{r} - \Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^3}
\qquad
\curl(\alpha\Vec{A})=(\grad\alpha)\cross\Vec{A}+\alpha\curl{\Vec{A}}
$$

可得

$$
\Vec{B}(\Vec{r})
= \frac{\mu_0}{4\pi}\int_{V(\Vec{r}')}\left(\grad\frac{1}{\vert\Vec{r}-\Vec{r}'\vert}\right)\cross\Vec{\jmath}(\Vec{r}')
= \curl\frac{\mu_0}{4\pi}\int_{V(\Vec{r}')}\frac{\Vec{\jmath}(\Vec{r}')}{\vert\Vec{r}-\Vec{r}'\vert} \eqqcolon \curl\Vec{A}(\Vec{r})
$$

其中 $ \Vec{A}(\Vec{r}) $ 被称为 $\Vec{B}(\Vec{r})$ 的**矢量势**。

### 静磁场的散度

利用微分恒等式

$$
\divg(\curl\Vec{A})=0\qquad\forall\Vec{A}
$$

可得

$$
\divg(\curl\Vec{A})=\boxed{\divg\Vec{B}=0}
$$

此即**真空中的稳态 Maxwell 方程组**的第三式。

### 静磁场的旋度

为计算 $\Vec{B}(\Vec{r})$ 的旋度，需利用微分恒等式

$$
\curl(\Vec{p}\cross\Vec{q})=(\grad\vdot\Vec{q}+\Vec{q}\vdot\grad)\Vec{p}-(\grad\vdot\Vec{p}+\Vec{p}\vdot\grad)\Vec{q}
$$

将其展开为四项，再利用

$$
\grad\frac{\Vec{r}-\Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^{3}}=-\grad'\frac{\Vec{r}-\Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^{3}}\impliedby(\forall f)\left(\grad f(\Vec{r}-\Vec{r}')=-\grad' f(\Vec{r}-\Vec{r}')\right)
$$

化简为

$$
\curl\Vec{B}(\Vec{r})=\frac{\mu_{0}}{4\pi}\int_{V(\Vec{r}')}\left(\divg\frac{\Vec{r}-\Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^{3}}\right)\Vec{\jmath}(\Vec{r}')
$$

将积分区域取为小球 $B(\Vec{r}',\delta)$ 并令 $\delta\to 0$ 即得

$$
\curl\Vec{B}(\Vec{r})
= \frac{\mu_0}{4\pi}\int_{B(\Vec{r}',\delta)}\Vec{\jmath}(\Vec{r}')\left(\grad\vdot\frac{\Vec{r} - \Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^3}\right)
\approx\Vec{\jmath}(\Vec{r})\oint_{\partial B(\Vec{r}',\delta)}\Vec{\nu}(\Vec{r}')\vdot\frac{\Vec{r}-\Vec{r}'}{\vert\Vec{r}-\Vec{r}'\vert^{3}}
$$

最终得

$$
\boxed{\curl\Vec{B}=\mu_0\Vec{\jmath}}
$$

此即**真空中的稳态 Maxwell 方程组**的第四式。

# 静电磁场

# 电磁波