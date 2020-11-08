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

### Lagrangian 作用量

### Lagrange's 方程

## 对称性 $\Rightarrow$ 守恒律

### 时间均匀性 $\Rightarrow$ 能量守恒

### 空间均匀性 $\Rightarrow$ 动量守恒

### 空间各向同性 $\Rightarrow$ 角动量守恒

# Hamiltonian 力学

## Hamilton's 方程

### Legendre's 变换

Lagrangian 作用量 $L(\Mat{q},\Mat{\dot{q}},t)$ 关于“广义速度”的导数被称为“广义动量”，即

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

利用 Legendre's 变换，可定义“Hamiltonian 作用量”

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

### 能量守恒

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

## Maupertuis' 原理

## Liouville's 定理

## Hamilton--Jacobi 方程

