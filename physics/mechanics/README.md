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

Lagrangian 作用量 $L(\Mat{q},\Mat{\dot{q}},t)$ 关于“广义速度”的导数

$$
\Mat{p}
\coloneqq\begin{bmatrix}p_1&\dots&p_n\end{bmatrix}
\coloneqq\begin{bmatrix}\dfrac{\partial L}{\partial q_1}&\dots&\dfrac{\partial L}{\partial q_n}\end{bmatrix}
\eqqcolon\pdv{}{\Mat{\dot{q}}}L(\Mat{q},\Mat{\dot{q}},t)
$$

被称为“广义动量”。

上述定义式右端是以 $(\Mat{q},\Mat{\dot{q}},t)$ 为自变量的（代数）表达式。
若将 $(\Mat{q},t)$ 及左端的 $\Mat{p}$ 视为已知量，而将 $\Mat{\dot{q}}$ 视为未知量，则该定义式可视为关于 $\Mat{\dot{q}}$ 的（代数）方程。
从中解得 $ \Mat{\dot{q}}=\mathopen{\Mat{\dot{q}}}(\Mat{q},\Mat{p},t) $，便可将 $L$ 也化作以 $(\Mat{q},\Mat{p},t)$ 为自变量的表达式。

## Poisson 括号

## Maupertuis' 原理

## Liouville's 定理

## Hamilton--Jacobi 方程

