---
title: 理论力学、分析力学
---

# Newtonian 力学

## 实验事实

### 惯性参考系

### Galileo's 相对性原理

**Galileo's 变换**：若惯性参考系 $K'$ 以速度 $\Vec{V}$ 相对于另一个惯性参考系 $K$ 运动，则同一质点在这两个参考系中的位置与时间满足如下关系

$$
\Vec{r}=\Vec{r}'+\Vec{V}t\qquad t=t'
$$

**Galileo's 相对性原理**：力学方程在 *Galileo's 变换*作用下具有不变性。

## 运动方程

### 质点

### 质点系

### 刚体

# Lagrangian 力学

## 最小作用量原理

### 广义坐标（速度）

$$
\Mat{q}\coloneqq\begin{bmatrix}q_1 & \dots & q_n\end{bmatrix}
$$

$$
\Mat{\dot{q}}
\coloneqq\begin{bmatrix}\dot{q}_1&\dots&\dot{q}_n\end{bmatrix}
\coloneqq\begin{bmatrix}\dfrac{\dd{q}_1}{\dd{t}}&\dots&\dfrac{\dd{q}_n}{\dd{t}}\end{bmatrix}
\eqqcolon\dv{}{t}\Mat{q}
$$

### 作用量的 Lagrangian 形式

$$
\boxed{S\coloneqq\int_{t_1}^{t_2}L(\Mat{q},\Mat{\dot{q}},t)\dd{t}}
$$

$$
\delta{S}=\int_{t_1}^{t_2}\delta{L}(\Mat{q},\Mat{\dot{q}},t)\dd{t}
=\int_{t_1}^{t_2}\left(\pdv{L}{\Mat{q}}\cdot\delta{\Mat{q}}+\pdv{L}{\Mat{\dot{q}}}\cdot\delta{\Mat{\dot{q}}}\right)\dd{t}
$$

### Lagrange's 方程<a href name="lagrange-eqn"></a>

$$
\int_{t_1}^{t_2}\pdv{L}{\Mat{\dot{q}}}\cdot\delta{\Mat{\dot{q}}}\dd{t}
=\left(\pdv{L}{\Mat{\dot{q}}}\cdot\delta{\Mat{q}}\right)_{t_1}^{t_2}
-\int_{t_1}^{t_2}\left(\dv{}{t}\pdv{L}{\Mat{\dot{q}}}\right)\cdot\delta{\Mat{q}}\dd{t}
$$

$$
\boxed{\dv{}{t}\pdv{L}{\Mat{\dot{q}}}=\pdv{L}{\Mat{q}}}
$$

【定理】若 $L_{*}(\underline{q},\underline{\dot{q}},t)$ 与 $L(\underline{q},\underline{\dot{q}},t)$ 只相差一个以 $\underline{q},t$ 为自变量的函数 $f(\underline{q},t)$ 关于 $t$ 的全导数，即 

$$
L_{*}(\underline{q},\underline{\dot{q}},t)=L(\underline{q},\underline{\dot{q}},t)+\dv{}{t}f(\underline{q},t)
$$

则它们给出相同的 *Lagrange's 方程*，从而在力学上完全等价。

## $L$ 的具体形式

### 自由质点

$$
L(\underline{q},\underline{\dot{q}},t)=L(v^{2})\qquad v^{2}\coloneqq\Vec{v}\vdot\Vec{v}
$$

$$
\boxed{L(\Vec{v})=\frac{m}{2}v^{2}}
$$

### 封闭质点系

$$
L(\Vec{r}_{1},\dots,\Vec{r}_{n},\Vec{v}_{1},\dots,\Vec{v}_{n},t)=\sum_{i=1}^{n}\frac{m_{i}}{2}\Vec{v}_{i}^{2}-V(\Vec{r}_{1},\dots,\Vec{r}_{n})
$$

$$
\boxed{L(\underline{q},\underline{\dot{q}},t)=\tfrac{1}{2}\underline{\dot{q}}\cdot\underline{A}(\underline{q})\cdot\underline{\dot{q}}-V(\underline{q})}
$$

### 外场的影响

单个质点：

$$
m\dv{v}{t}=-\pdv{V}{\Vec{r}}\impliedby L=\frac{1}{2}mv^{2}-V(\Vec{r},t)
$$

质点系：

$$
L(\Vec{r}_{1},\dots,\Vec{r}_{n},\Vec{v}_{1},\dots,\Vec{v}_{n},t)=\sum_{i=1}^{n}\frac{m_{i}}{2}\Vec{v}_{i}^{2}-V(\Vec{r}_{1},\dots,\Vec{r}_{n},t)
$$

## 对称性 $\Rightarrow$ 守恒律

### 时间均匀性 $\Rightarrow$ 能量守恒

$$
\dv{L(\underline{q},\underline{\dot{q}})}{t}=\pdv{L}{\underline{q}}\cdot\dv{\underline{q}}{t}+\pdv{L}{\underline{\dot{q}}}\cdot\dv{\underline{\dot{q}}}{t}
=\dv{}{t}\left(\pdv{L}{\underline{\dot{q}}}\cdot\dv{\underline{q}}{t}\right)
$$

$$
\boxed{E\coloneqq\pdv{L}{\underline{\dot{q}}}\cdot\underline{\dot{q}}-L=\text{const}}
$$

### 空间均匀性 $\Rightarrow$ 动量守恒

全空间的任意无穷小平移 $\delta\Vec{r}\eqqcolon\Vec{\epsilon}$ 不改变系统的力学行为，即

$$
0=\delta L=\sum_{i=1}^{n}\pdv{L}{\Vec{r}_{i}}\vdot\delta\Vec{r}_{i}=\Vec{\epsilon}\vdot\sum_{i=1}^{n}\pdv{L}{\Vec{r}_{i}}=\Vec{\epsilon}\vdot\dv{}{t}\sum_{i=1}^{n}\pdv{L}{\Vec{v}_{i}}
$$

$$
\boxed{\Vec{p}\coloneqq\sum_{i=1}^{n}\Vec{p}_{i}\coloneqq\sum_{i=1}^{n}\pdv{L}{\Vec{v}_{i}}=\sum_{i=1}^{n}m_{i}\Vec{v}_{i}=\text{const}}
$$

### 空间各向同性 $\Rightarrow$ 角动量守恒

全空间的任意无穷小旋转 $\delta\Vec{\varphi}$ 不改变系统的力学行为，即

$$
\begin{aligned}0=\delta L & =\sum_{i=1}^{n}\pdv{L}{\Vec{r}_{i}}\vdot\delta\Vec{r}_{i}+\sum_{i=1}^{n}\pdv{L}{\Vec{v}_{i}}\vdot\delta\Vec{v}_{i}\\
 & =\sum_{i=1}^{n}\dot{\Vec{p}}_{i}\vdot\left(\delta\Vec{\varphi}\cross\Vec{r}_{i}\right)+\sum_{i=1}^{n}\Vec{p}_{i}\vdot\left(\delta\Vec{\varphi}\cross\Vec{v}_{i}\right)\\
 & =\delta\Vec{\varphi}\vdot\sum_{i=1}^{n}\dv{}{t}\left(\Vec{r}_{i}\cross\Vec{p}_{i}\right)\eqqcolon\delta\Vec{\varphi}\vdot\dv{}{t}\sum_{i=1}^{n}\Vec{L}_{i}
\end{aligned}
$$

$$
\boxed{\Vec{L}\coloneqq\sum_{i=1}^{n}\Vec{L}_{i}\coloneqq\sum_{i=1}^{n}\Vec{r}_{i}\cross\Vec{p}_{i}=\text{const}}
$$

# Hamiltonian 力学

## Hamilton's 方程

### Legendre's 变换

Lagrangian 函数 $L(\Mat{q},\Mat{\dot{q}},t)$ *关于广义速度的导数*被称为**广义动量**，即

$$
\Mat{p}
\coloneqq\begin{bmatrix}p_1&\dots&p_n\end{bmatrix}
\coloneqq\begin{bmatrix}\dfrac{\partial L}{\partial\dot{q}_1}&\dots&\dfrac{\partial L}{\partial\dot{q}_n}\end{bmatrix}
\eqqcolon\pdv{}{\Mat{\dot{q}}}L(\Mat{q},\Mat{\dot{q}},t)
$$

上述定义式右端是以 $(\Mat{q},\Mat{\dot{q}},t)$ 为自变量的（代数）表达式。
若将 $(\Mat{q},t)$ 及左端的 $\Mat{p}$ 视为已知量，而将 $\Mat{\dot{q}}$ 视为未知量，则该定义式可视为关于 $\Mat{\dot{q}}$ 的（代数）方程。
从中解得 $ \Mat{\dot{q}}=\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t) $，便可将 $L(\Mat{q},\Mat{\dot{q}},t)$ 也化作以 $(\Mat{q},\Mat{p},t)$ 为自变量的表达式，即

$$
\tilde{L}(\Mat{q},\Mat{p},t)\coloneqq \mathopen{L}\left(\Mat{q},\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t),t\right)
$$

由此可定义 **Hamiltonian 函数**

$$
\boxed{H(\Mat{q},\Mat{p},t)\coloneqq \Mat{p}\cdot\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t)- \tilde{L}(\Mat{q},\Mat{p},t)}
$$

其物理意义为系统的能量。
上述由 $L(\Mat{q},\Mat{\dot{q}},t)$ 导出 $H(\Mat{q},\Mat{p},t)$ 的过程在数学上被称为 **Legendre's 变换**

### Hamilton's 方程

$H(\Mat{q},\Mat{p},t)$ 的全微分可写成

$$
\dd{H}
=\dd{(\Mat{p}\cdot\Mat{\dot{q}}-\tilde{L})}
=\Mat{p}\cdot\dd{\Mat{\dot{q}}}+\Mat{\dot{q}}\cdot\dd{\Mat{p}}
-\pdv{L}{\Mat{q}}\cdot\dd{\Mat{q}}-\pdv{L}{\Mat{\dot{q}}}\cdot\dd{\Mat{\dot{q}}}-\pdv{L}{t}\dd{t}
$$

利用 $\Mat{p}$ 的定义及 [Lagrange's 方程](#lagrange-eqn)，可得

$$
\dd{H}
=\pdv{H}{\Mat{p}}\cdot\dd{\Mat{p}}
+\pdv{H}{\Mat{q}}\cdot\dd{\Mat{q}}
+\pdv{H}{t}\dd{t}
=\Mat{\dot{q}}\cdot\dd{\Mat{p}}
-\Mat{\dot{p}}\cdot\dd{\Mat{q}}
-\pdv{L}{t}\dd{t}
$$

比较后一个等号两侧 $\dd{\Mat{p}},\dd{\Mat{q}}$ 系数，即得 **Hamilton's 方程**<a href name="hamilton-eqn"></a>

$$
\boxed{\Mat{\dot{q}}=+\pdv{H}{\Mat{p}}\qquad\Mat{\dot{p}}=-\pdv{H}{\Mat{q}}}
$$

此方程只含一阶导数，且具有很好的对称性，在分析力学中居于核心地位，故又名**正则方程 (canonical equations)**。

### 能量守恒<a href name="const-H"></a>

比较 $\dd{t}$ 两侧的系数，可得

$$
\boxed{\pdv{H}{t}=-\pdv{L}{t}}
$$

其中表示*时间*的变量 $t$ 可以推广为除了 $p,q$ 以外的任何决定 $L,H$ 的参数。

利用 [Hamilton's 方程](#hamilton-eqn)，可以将*能量守恒*条件化为

$$
\dv{H}{t}
=\cancel{\Mat{\dot{q}}\cdot\dv{\Mat{p}}{t}-\Mat{\dot{p}}\cdot\dv{\Mat{q}}{t}}
+\boxed{\pdv{H}{t}=0}
$$

即 *$H$ 不显含时间*。

## 作用量的 Hamiltonian 形式

### 边界条件的作用

真实运动所对应的作用量 $S=\int_{t_1}^{t_2} L(\Mat{q},\Mat{\dot{q}},t)\dd{t}$ 可以被视为由始末时刻 $t_1,t_2$ 及位置 $\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2)$ 所确定的量，即

$$
S=\mathopen{S}\left(\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2),t_1,t_2\right)
$$

若固定 $t_1,t_2$ 但允许 $\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2)$ 变化，则

$$
\delta{S}
=\pdv{S}{\mathopen{\Mat{q}}(t_2)}\cdot\mathopen{\delta}\mathopen{\Mat{q}}(t_2)
+\pdv{S}{\mathopen{\Mat{q}}(t_1)}\cdot\mathopen{\delta}\mathopen{\Mat{q}}(t_1)
=\int_{t_1}^{t_2}\left(\pdv{L}{\Mat{q}}-\dv{}{t}\pdv{L}{\Mat{\dot{q}}}\right)\cdot\mathopen{\delta}\Mat{q}\dd{t}
+\left(\pdv{L}{\Mat{\dot{q}}}\cdot\mathopen{\delta}\Mat{q}\right)_{t_1}^{t_2}
$$

真实轨道应满足 [Lagrange's 方程](#lagrange-eqn)，故被积函数为零；边界项中的偏导数可以用广义动量替换，因此有

$$
\delta{S}
=\mathopen{\Mat{p}}(t_2)\cdot\mathopen{\delta}\mathopen{\Mat{q}}(t_2)
-\mathopen{\Mat{p}}(t_1)\cdot\mathopen{\delta}\mathopen{\Mat{q}}(t_1)
$$

比较系数，即得

$$
\mathopen{\Mat{p}}(t_2)=+\pdv{S}{\mathopen{\Mat{q}}(t_2)}\qquad
\mathopen{\Mat{p}}(t_1)=-\pdv{S}{\mathopen{\Mat{q}}(t_1)}
$$

将上式代入*关于 $t_2,t_1$ 全导数*（依次固定 $t_1,t_2$ 并利用变上、下限积分的导数公式）

$$
\dv{S}{t_2}=+L(t_2)=\pdv{S}{t_2}+\pdv{S}{\mathopen{\Mat{q}}(t_2)}\cdot\mathopen{\Mat{\dot{q}}}(t_2)\qquad
\dv{S}{t_1}=-L(t_1)=\pdv{S}{t_1}+\pdv{S}{\mathopen{\Mat{q}}(t_1)}\cdot\mathopen{\Mat{\dot{q}}}(t_1)
$$

并利用 $H=\Mat{p}\cdot\Mat{\dot{q}}-L$，可得

$$
\pdv{S}{t_2}=-H(t_2)\qquad\pdv{S}{t_1}=+H(t_1)
$$

将它们代回*关于 $t_2,t_1$ 全导数*，即得

$$
\dv{S}{t_2}=+\mathopen{\Mat{p}}(t_2)\cdot\mathopen{\Mat{\dot{q}}}(t_2)-H(t_2)\qquad
\dv{S}{t_1}=-\mathopen{\Mat{p}}(t_1)\cdot\mathopen{\Mat{\dot{q}}}(t_1)+H(t_1)
$$

作用量 $\mathopen{S}\left(\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2),t_1,t_2\right)$ 的全微分（允许 $t_1,t_2$ 一起变化）应当是这两个微商与各自时间微元的乘积之和，即

$$
\boxed{\mathopen{\dd{S}}\left(\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2),t_1,t_2\right)
=\left(\mathopen{\Mat{p}}(t)\cdot\mathopen{\dd{\Mat{q}}}(t)-H(t)\dd{t}\right)_{t_1}^{t_2}}
$$

该式表明：真实轨道必须使右端表达式为全微分。

### 微分与积分形式

特别地，若取定初始位置（即 $\mathopen{\delta}\mathopen{\Mat{q}}(t_1)=0$）并省略 $t_2$ 的下标，则有

$$
\dd{S}=\Mat{p}\cdot\dd{\Mat{q}}-H\dd{t}\implies
\boxed{S=\int\left(\Mat{p}\cdot\dd{\Mat{q}}-H\dd{t}\right)}
$$

将 $\Mat{p},\Mat{q}$ 视为 $S$ 的独立变量，利用最小作用量原理 $\delta S=0$，可重新导出 [Hamilton's 方程](#hamilton-eqn)。

### Maupertuis' 原理

只考虑满足[能量守恒](#const-H) $ H(\Mat{p},\Mat{q})=E $ 的系统。
若取定初始时刻 $t_0$ 及始末位置 $\mathopen{\Mat{q}}(t_0),\mathopen{\Mat{q}}(t)$ 且允许终止时刻 $t$ 变化，则有

$$
\delta{S}=-H(\Mat{p},\Mat{q})\,\delta{t}=-E\,\delta{t}
$$

另一方面，对作用量的 Hamiltonian 形式

$$
S=-(t-t_0)E+\int_{t_0}^t\Mat{p}\cdot\dd{\Mat{q}}
$$

作变分，可得

$$
\delta{S}=-E\,\delta{t}+\delta{S_0}\qquad S_0\coloneqq\int_{t_0}^t\Mat{p}\cdot\dd{\Mat{q}}
$$

其中 $S_0$ 被称为**简约作用量**。消去 $\delta{S}=-E\,\delta{t}$，就得到 **Maupertuis' 原理**

$$
\delta{S_0}\equiv\boxed{\delta{\int_{t_0}^t\Mat{p}\cdot\dd{\Mat{q}}}=0}
$$

基于该原理，可以解出**轨道**，即不含时间的曲线方程。具体做法如下：

1. 由 $E=E(\Mat{q},\Mat{\dot{q}})$ 解出 $\dd{t}$，将 $\dd{t}$ 表示成以 $\Mat{q},\dd{\Mat{q}}$ 为自变量的函数。
2. 将上述 $\dd{t}$ 代入 $\Mat{p}=\partial L/\mathopen{\partial}\Mat{\dot{q}}$，将 $\Mat{p}$ 也表示成以 $\Mat{q},\dd{\Mat{q}}$ 为自变量的函数。
3. 将上述 $\Mat{p}$ 代入 $\delta{S_0}=0$，即得 $\Mat{q},\dd{\Mat{q}}$ 所应满足的方程，此即**轨道方程**。

**例**：对于典型系统

$$
L=\tfrac12\Mat{\dot{q}}\cdot\mathopen{\Mat{A}}(\Mat{q})\cdot\Mat{\dot{q}}-U(\Mat{q})\qquad
\Mat{p}=\Mat{\dot{q}}\cdot\mathopen{\Mat{A}}(\Mat{q})\qquad
E=\tfrac12\Mat{\dot{q}}\cdot\mathopen{\Mat{A}}(\Mat{q})\cdot\Mat{\dot{q}}+U(\Mat{q})
$$

上述步骤给出

$$
\begin{aligned}
\dd{t}=\sqrt{\frac{\Mat{q}\cdot\mathopen{\Mat{A}}(\Mat{q})\cdot\Mat{q}}{2(E-U(\Mat{q}))}}
&\implies
\Mat{p}\cdot\dd{\Mat{q}}
=\frac{\dd{\Mat{q}}\cdot\mathopen{\Mat{A}}(\Mat{q})\cdot\dd{\Mat{q}}}{\dd{t}}
=\sqrt{2(E-U(\Mat{q}))\dd{\Mat{q}}\cdot\mathopen{\Mat{A}}(\Mat{q})\cdot\dd{\Mat{q}}}\\
&\implies\boxed{\delta\int_{t_0}^t\sqrt{2(E-U(\Mat{q}))\dd{\Mat{q}}\cdot\mathopen{\Mat{A}}(\Mat{q})\cdot\dd{\Mat{q}}}=0}
\end{aligned}
$$

## 正则变换

### 正则变换条件

一般的变量替换

$$
\Mat{\tilde{q}}=\mathopen{\Mat{\tilde{q}}}(\Mat{q},\Mat{p},t)\qquad
\Mat{\tilde{p}}=\mathopen{\Mat{\tilde{p}}}(\Mat{q},\Mat{p},t)
$$

不能保证新的运动方程仍具有正则形式，即

$$
\Mat{\dot{\tilde{q}}}=+\pdv{\tilde{H}}{\Mat{\tilde{p}}}\qquad\Mat{\dot{\tilde{p}}}=-\pdv{\tilde{H}}{\Mat{\tilde{q}}}
$$

但满足如下条件

$$
\boxed{(\exists F)\left(\dd{F}=\dd{S}-\dd{\tilde{S}}\right)}\qquad
\begin{cases}
\dd{S}=\Mat{p}\cdot\dd{\Mat{q}}-H\dd{t}\\
\dd{\tilde{S}}=\Mat{\tilde{p}}\cdot\dd{\Mat{\tilde{q}}}-\tilde{H}\dd{t}
\end{cases}
$$

的变换能够保持方程的正则性，因此该条件被称为**正则变换条件**，其中 $F=F(\Mat{q},\Mat{p},\Mat{\tilde{q}},\Mat{\tilde{p}},t)$ 被称为该变换的**生成函数**。
这是因为，变换前后的作用量 $S,\tilde{S}$ 能分别导出 Hamilton's 方程，并且二者之差 $S-\tilde{S}=\left.F\right|_{t_1}^{t_2}$ 为不影响变分的常数。

### 正则变换公式

将正则变换条件

$$
\dd{F}
=\Mat{p}\cdot\dd{\Mat{q}}
-\Mat{\tilde{p}}\cdot\dd{\Mat{\tilde{q}}}
+(\tilde{H}-H)\dd{t}
$$

与全微分

$$
\dd{F}
=\pdv{F}{\Mat{q}}\cdot\dd{\Mat{q}}
+\pdv{F}{\Mat{\tilde{q}}}\cdot\dd{\Mat{\tilde{q}}}
+\pdv{F}{t}\dd{t}
$$

比较系数，可得第一种正则变换公式：

$$
\boxed{\Mat{p}=\pdv{F}{\Mat{q}}\qquad
\Mat{\tilde{p}}=-\pdv{F}{\Mat{\tilde{q}}}\qquad
\tilde{H}-H=\pdv{F}{t}}
$$

其中生成函数 $F$ 只依赖于 $\Mat{q},\Mat{p},t$；若要获得只依赖于 $\Mat{q},\Mat{\tilde{p}},t$ 的生成函数，则需对正则变换条件作 *Legendre's 变换*：

$$
\dd{\varPhi}\coloneqq\dd{(F+\Mat{\tilde{p}}\cdot\Mat{\tilde{q}})}
=\Mat{p}\cdot\dd{\Mat{q}}
+\Mat{\tilde{q}}\cdot\dd{\Mat{\tilde{p}}}
+(\tilde{H}-H)\dd{t}
$$

并与全微分

$$
\dd{\varPhi}
=\pdv{\varPhi}{\Mat{q}}\cdot\dd{\Mat{q}}
+\pdv{\varPhi}{\Mat{\tilde{p}}}\cdot\dd{\Mat{\tilde{p}}}
+\pdv{\varPhi}{t}\dd{t}
$$

比较系数，由此得第二种正则变换公式：

$$
\boxed{\Mat{p}=\pdv{\varPhi}{\Mat{q}}\qquad
\Mat{\tilde{q}}=\pdv{\varPhi}{\Mat{\tilde{p}}}\qquad
\tilde{H}-H=\pdv{\varPhi}{t}}
$$

类似地，可得另外两种母函数及相应的正则变换公式：

$$
\dd{G}\coloneqq\dd{(F-\underline{p}\cdot\underline{q})}=-\underline{q}\cdot\dd{\underline{p}}-\underline{\tilde{p}}\cdot\dd{\underline{\tilde{q}}}+(\tilde{H}-H)\dd{t}
$$

$$
\boxed{\underline{q}=-\pdv{G}{\underline{p}}\qquad\underline{\tilde{p}}=-\pdv{G}{\underline{\tilde{q}}}\qquad\tilde{H}-H=\pdv{G}{t}}
$$

$$
\dd{\varPsi}\coloneqq\dd{(\varPhi-\underline{p}\cdot\underline{q})}=-\underline{q}\cdot\dd{\underline{p}}+\underline{\tilde{q}}\cdot\dd{\underline{\tilde{p}}}+(\tilde{H}-H)\dd{t}
$$

$$
\boxed{\underline{q}=-\pdv{\varPsi}{\underline{p}}\qquad\underline{\tilde{q}}=\pdv{\varPsi}{\underline{\tilde{p}}}\qquad\tilde{H}-H=\pdv{\varPsi}{t}}
$$


### 正则共轭变量

### Liouville's 定理

【引理】真实运动所引起的正则共轭变量 $\underline{q},\underline{p}$ 的变化，可以看作一系列正则变换累加的结果。

**Liouville's 定理**：相空间中任意点集的测度不随这些点的（满足力学定律的真实）运动而变化。

## Poisson 括号

### 定义

给定两个依赖于 $(\Mat{p},\Mat{q})$ 的函数 $f(\Mat{p},\Mat{q}),g(\Mat{p},\Mat{q})$，它们的 **Poisson 括号**是指

$$
\boxed{\{f,g\}\coloneqq\pdv{f}{\Mat{p}}\cdot\pdv{g}{\Mat{q}}-\pdv{f}{\Mat{q}}\cdot\pdv{g}{\Mat{p}}}
$$

于是 $f(\Mat{p},\Mat{q})$ 关于 $t$ 的全导数可以被改写为

$$
\dv{f}{t}
=\pdv{f}{t}+\pdv{f}{\Mat{p}}\cdot\Mat{\dot{p}}+\pdv{f}{\Mat{q}}\cdot\Mat{\dot{q}}
=\pdv{f}{t}-\pdv{f}{\Mat{p}}\cdot\pdv{H}{\Mat{q}}+\pdv{f}{\Mat{q}}\cdot\pdv{H}{\Mat{p}}
=\pdv{f}{t}+\{H,f\}
$$

⚠️ 某些文献将上述定义中的 $p,q$ 互换，所得结果与这里正好相差一个负号。这种差别不是实质性的，只要上下文保持一致即可。

### 恒等式

$$
\{f,g\}=\{g,f\}\qquad\{f,1\}=0
$$

$$
\{f_1+f_2,g\}=\{f_1,g\}+\{f_2,g\}\qquad\{f,g_1+g_2\}=\{f,g_1\}+\{f,g_2\}
$$

$$
\{f_1f_2,g\}=f_1\{f_2,g\}+\{f_1,g\}f_2\qquad\{f,g_1g_2\}=g_1\{f,g_2\}+\{f,g_1\}g_2
$$

$$
\{f,\{g,h\}\}+\{g,\{h,f\}\}+\{h,\{f,g\}\}=0
$$

$$
\{f,q_i\}=\pdv{f}{p_i}\qquad\{p_i,f\}=\pdv{f}{q_i}\qquad
\{q_i,q_k\}=0\qquad\{p_i,p_k\}=0\qquad\{p_i,q_k\}=\delta_{ik}
$$

### 运动积分

**定理**：若 $f(\Mat{p},\Mat{q}),g(\Mat{p},\Mat{q})$ 均为运动积分，则 $\{f,g\}$ 亦为运动积分，即

$$
\left(\dv{f}{t}=0\right)\land\left(\dv{g}{t}=0\right)\implies\dv{}{t}\{f,g\}=0
$$

### 正则变换条件

$$
\{f,g\}_{\underline{p},\underline{q}}=\{f,g\}_{\underline{\tilde{p}},\underline{\tilde{q}}}
$$

$$
\{\tilde{q}_{i},\tilde{q}_{k}\}_{\underline{p},\underline{q}}=0\qquad\{\tilde{p}_{i},\tilde{p}_{k}\}_{\underline{p},\underline{q}}=0\qquad\{\tilde{p}_{i},\tilde{q}_{k}\}_{\underline{p},\underline{q}}=\delta_{ik}
$$

## Hamilton--Jacobi 方程

$$
\boxed{\frac{\partial S}{\partial t}+\mathopen{H}\left(\underline{q},\frac{\partial S}{\partial\underline{q}},t\right)=0}
$$
