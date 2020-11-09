---
title: 理论力学
---

# Newtonian 力学

## 实验事实

### Galileo's 相对性原理

## 运动方程

# Lagrangian 力学

## 最小作用量原理

### 广义坐标（速度）

### Lagrangian 函数

### Lagrange's 方程<a href name="lagrange-eqn"></a>

## 对称性 $\Rightarrow$ 守恒律

### 时间均匀性 $\Rightarrow$ 能量守恒

### 空间均匀性 $\Rightarrow$ 动量守恒

### 空间各向同性 $\Rightarrow$ 角动量守恒

# Hamiltonian 力学

## Hamilton's 方程

### Legendre's 变换

Lagrangian 函数 $L(\Mat{q},\Mat{\dot{q}},t)$ 关于“广义速度”的导数被称为“广义动量”，即

$$
\Mat{p}
\coloneqq\begin{bmatrix}p_1&\dots&p_n\end{bmatrix}
\coloneqq\begin{bmatrix}\dfrac{\partial L}{\partial q_1}&\dots&\dfrac{\partial L}{\partial q_n}\end{bmatrix}
\eqqcolon\pdv{}{\Mat{\dot{q}}}L(\Mat{q},\Mat{\dot{q}},t)
$$

上述定义式右端是以 $(\Mat{q},\Mat{\dot{q}},t)$ 为自变量的（代数）表达式。
若将 $(\Mat{q},t)$ 及左端的 $\Mat{p}$ 视为已知量，而将 $\Mat{\dot{q}}$ 视为未知量，则该定义式可视为关于 $\Mat{\dot{q}}$ 的（代数）方程。
从中解得 $ \Mat{\dot{q}}=\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t) $，便可将 $L(\Mat{q},\Mat{\dot{q}},t)$ 也化作以 $(\Mat{q},\Mat{p},t)$ 为自变量的表达式，即
$$
\tilde{L}(\Mat{q},\Mat{p},t)\coloneqq \mathopen{L}\left(\Mat{q},\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t),t\right)
$$

利用 Legendre's 变换，可定义“Hamiltonian 函数”

$$
\boxed{H(\Mat{q},\Mat{p},t)\coloneqq \Mat{p}\cdot\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t)- \tilde{L}(\Mat{q},\Mat{p},t)}
$$

其物理意义为系统的能量。

### Hamilton's 方程

$H(\Mat{q},\Mat{p},t)$ 的全微分可写成

$$
\dd{H}
=\dd{(\Mat{p}\cdot\Mat{\dot{q}}-\tilde{L})}
=\Mat{p}\cdot\dd{\Mat{\dot{q}}}+\Mat{\dot{q}}\cdot\dd{\Mat{p}}
-\pdv{L}{\Mat{q}}\cdot\dd{\Mat{q}}-\pdv{L}{\Mat{\dot{q}}}\cdot\dd{\Mat{\dot{q}}}-\pdv{L}{t}\dd{t}
$$

利用 $\Mat{p}$ 的定义及 Lagrange's 方程

$$
\Mat{\dot{p}}\equiv\dv{}{t}\pdv{L}{\Mat{\dot{q}}}=\pdv{L}{\Mat{q}}
$$

可得

$$
\dd{H}
=\pdv{H}{\Mat{p}}\cdot\dd{\Mat{p}}
+\pdv{H}{\Mat{q}}\cdot\dd{\Mat{q}}
+\pdv{H}{t}\dd{t}
=\Mat{\dot{q}}\cdot\dd{\Mat{p}}
-\Mat{\dot{p}}\cdot\dd{\Mat{q}}
-\pdv{L}{t}
$$

比较后一个等号两侧 $\dd{\Mat{p}},\dd{\Mat{q}}$ 系数，即得“Hamilton's 方程”

$$
\boxed{\Mat{\dot{q}}=+\pdv{H}{\Mat{p}}\qquad\Mat{\dot{p}}=-\pdv{H}{\Mat{q}}}
$$

此方程只含一阶导数，且具有很好的对称性，在分析力学中居于核心地位，故又名“正则方程 (canonical equations)”。

### 能量守恒<a href name="const-H"></a>

比较 $\dd{t}$ 两侧的系数，可得

$$
\boxed{\pdv{H}{t}=-\pdv{L}{t}}
$$

其中表示“时间”的变量 $t$ 可以推广为除 $p,q$ 以外的决定 $L,H$ 的参数。

利用 Hamilton's 方程，可以将“能量守恒”条件化为

$$
\dv{H}{t}
=\cancel{\Mat{\dot{q}}\cdot\dv{\Mat{p}}{t}-\Mat{\dot{p}}\cdot\dv{\Mat{q}}{t}}
+\boxed{\pdv{H}{t}=0}
$$

即“$H$ 不显含时间”。

## Poisson 括号

### 定义

给定两个依赖于 $(\Mat{p},\Mat{q})$ 的函数 $f(\Mat{p},\Mat{q}),g(\Mat{p},\Mat{q})$，它们的“Poisson 括号”是指

$$
\boxed{\{f,g\}\coloneqq\pdv{f}{\Mat{p}}\pdv{g}{\Mat{q}}-\pdv{f}{\Mat{q}}\pdv{g}{\Mat{p}}}
$$

于是 $f(\Mat{p},\Mat{q})$ 关于 $t$ 的全导数可以被改写为

$$
\dv{f}{t}
=\pdv{f}{t}+\pdv{f}{\Mat{p}}\Mat{\dot{p}}+\pdv{f}{\Mat{q}}\Mat{\dot{q}}
=\pdv{f}{t}-\pdv{f}{\Mat{p}}\pdv{H}{\Mat{q}}+\pdv{f}{\Mat{q}}\pdv{H}{\Mat{p}}
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

定理：若 $f(\Mat{p},\Mat{q}),g(\Mat{p},\Mat{q})$ 均为运动积分，则 $\{f,g\}$ 亦为运动积分，即

$$
\left(\dv{f}{t}=0\right)\land\left(\dv{g}{t}=0\right)\implies\dv{}{t}\{f,g\}=0
$$

## 作用量的 Hamiltonian 形式

### 边界条件的作用

作用量 $S=\int_{t_1}^{t_2} L(\Mat{q},\Mat{\dot{q}},t)\dd{t}$ 可以被视为由端点 $\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2)$ 及连接二者的真实轨道所确定的量，即

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

将上式代入“全导数”（依次固定 $t_1,t_2$ 并利用变上、下限积分的导数公式）

$$
\dv{S}{t_2}=+L(t_2)=\pdv{S}{t_2}+\pdv{S}{\mathopen{\Mat{q}}(t_2)}\cdot\mathopen{\Mat{\dot{q}}}(t_2)\qquad
\dv{S}{t_1}=-L(t_1)=\pdv{S}{t_1}+\pdv{S}{\mathopen{\Mat{q}}(t_1)}\cdot\mathopen{\Mat{\dot{q}}}(t_1)
$$

并利用 $H=\Mat{p}\cdot\Mat{\dot{q}}-L$，可得

$$
\pdv{S}{t_2}=-H(t_2)\qquad\pdv{S}{t_1}=+H(t_1)
$$

将它们代回“全导数”，即得

$$
\dv{S}{t_2}=+\mathopen{\Mat{p}}(t_2)\cdot\mathopen{\Mat{\dot{q}}}(t_2)-H(t_2)\qquad
\dv{S}{t_1}=-\mathopen{\Mat{p}}(t_1)\cdot\mathopen{\Mat{\dot{q}}}(t_1)+H(t_1)
$$

作用量 $\mathopen{S}\left(\mathopen{\Mat{q}}(t_1),\mathopen{\Mat{q}}(t_2),t_1,t_2\right)$ 的全微分（允许 $t_1,t_2$ 一起变化），应当是这两个微商与各自时间微元的乘积之和，即

$$
\boxed{\dd{S}
=\left(\mathopen{\Mat{p}}(t_2)\cdot\mathopen{\dd{\Mat{q}}}(t_2)-H(t_2)\dd{t_2}\right)
-\left(\mathopen{\Mat{p}}(t_1)\cdot\mathopen{\dd{\Mat{q}}}(t_1)-H(t_1)\dd{t_1}\right)}
$$

该式表明：右端表达式为全微分的轨道才有可能对应真实运动。

### 微分与积分形式

特别地，若取定初始状态（即 $\mathopen{\delta}\mathopen{\Mat{q}}(t_1)=0$）并省略 $t_2$ 的下标，则有

$$
\dd{S}=\Mat{p}\cdot\dd{\Mat{q}}-H\dd{t}\implies
\boxed{S=\int\left(\Mat{p}\cdot\dd{\Mat{q}}-H\dd{t}\right)}
$$

将 $\Mat{p},\Mat{q}$ 视为 $S$ 的独立变量，利用最小作用量原理 $\delta S=0$，可重新导出 Hamilton's 方程。

### Maupertuis' 原理

只考虑满足[能量守恒](#const-H) $ H(\Mat{p},\Mat{q})=E=\text{const} $ 的系统。
若取定初始时刻 $t_0$ 及始末位置 $\mathopen{\Mat{q}}(t_0),\mathopen{\Mat{q}}(t)$，且允许终止时刻 $t$ 变化，则有
$$
\delta{S}=-H(\Mat{p},\Mat{q})\,\delta{t}=-E\,\delta{t}
$$

另一方面，对作用量的 Hamiltonian 形式

$$
S=-E(t-t_0)+\int_{t_0}^t\Mat{p}\cdot\dd{\Mat{q}}
$$

作变分，可得

$$
\delta{S}=-E\,\delta{t}+\delta{S_0}\qquad S_0\coloneqq\int_{t_0}^t\Mat{p}\cdot\dd{\Mat{q}}
$$

其中 $S_0$ 被称为“简约作用量”。消去相等的项，就得到“Maupertuis' 原理”

$$
\delta{S_0}\equiv\boxed{\delta{\int_{t_0}^t\Mat{p}\cdot\dd{\Mat{q}}}=0}
$$

基于该原理，可以解出“轨道”，即不含时间的曲线方程。具体做法如下：

1. 由 $E=E(\Mat{q},\Mat{\dot{q}})$ 解出 $\dd{t}$，将 $\dd{t}$ 表示成以 $\Mat{q},\dd{\Mat{q}}$ 为自变量的函数。
2. 将上述 $\dd{t}$ 代入 $\Mat{p}=\partial L/\mathopen{\partial}\Mat{\dot{q}}$，将 $\Mat{p}$ 也表示成以 $\Mat{q},\dd{\Mat{q}}$ 为自变量的函数。
3. 将上述 $\Mat{p}$ 代入 $\delta{S_0}=0$，即得 $\Mat{q},\dd{\Mat{q}}$ 所应满足的方程，此即“轨道方程”。

例：对于典型系统

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

### Liouville's 定理

## Hamilton--Jacobi 方程

