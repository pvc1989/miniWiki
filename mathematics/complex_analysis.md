---
title: 复变函数论、复分析
---

# 复数

## 定义

### 有序实数对

$$
\mathbb{C}\coloneqq\left\{ (x,y):(x,y)\in\mathbb{R}^{2}\right\} 
$$

$$
(x_{1},y_{1})+(x_{2},y_{2})\coloneqq(x_{1}+x_{2},\ y_{1}+y_{2})
$$

$$
(x_{1},y_{1})\times(x_{2},y_{2})\coloneqq(x_{1}x_{2}-y_{1}y_{2},\ x_{1}y_{2}+x_{2}y_{1})
$$

$$
1\equiv(1,0),\quad
\sqrt{-1}\equiv(0,1)
$$

### 反对称实数矩阵

$$
\mathbb{C}\coloneqq\left\{ \begin{bmatrix}x & y\\
-y & x
\end{bmatrix}:(x,y)\in\mathbb{R}^{2}\right\} 
$$

$$
\begin{bmatrix}x_{1} & y_{1}\\
-y_{1} & x_{1}
\end{bmatrix}+\begin{bmatrix}x_{2} & y_{2}\\
-y_{2} & x_{2}
\end{bmatrix} \coloneqq\begin{bmatrix}x_{1}+x_{2} & y_{1}+y_{2}\\
-y_{1}-y_{2} & x_{1}+x_{2}
\end{bmatrix}
$$

$$
\begin{bmatrix}x_{1} & y_{1}\\
-y_{1} & x_{1}
\end{bmatrix}\times\begin{bmatrix}x_{2} & y_{2}\\
-y_{2} & x_{2}
\end{bmatrix} \coloneqq\begin{bmatrix}x_{1}x_{2}-y_{1}y_{2} & x_{1}y_{2}+x_{2}y_{1}\\
-x_{1}y_{2}-x_{2}y_{1} & x_{1}x_{2}-y_{1}y_{2}
\end{bmatrix}
$$

$$
1\equiv\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix},\quad
\sqrt{-1}\equiv\begin{bmatrix}0 & 1\\-1 & 0\end{bmatrix}
$$

## 几何表示

### 直角坐标

$$
z=x+\sqrt{-1}y,\quad(x,y)\in\mathbb{R}^{2}
$$

- 实部 $$ \mathrm{Re}(z)\coloneqq x $$
- 虚部 $$ \mathrm{Im}(z)\coloneqq y $$
- 复共轭 $$ z^* \coloneqq x - \sqrt{-1}y $$

### 极坐标

$$
z=\rho\cos\theta+\sqrt{-1}\rho\sin\theta,\quad(\rho,\theta)\in[0,\infty)\times(-\infty,\infty)
$$

- 模 $$ \rho\coloneqq\sqrt{zz^*} $$
- 辐角 $$ \theta \coloneqq \arctan(y/x) $$
- 辐角主值 $$ \theta\in[0,2\mathrm{\pi}) $$

### 复平面

复平面默认不含无穷远点，即

$$
\mathbb{C}\coloneqq\left\{ x+\sqrt{-1}y:\vert x\vert+\vert y\vert<\infty\right\} 
$$

### 复球面

以复平面为赤道面（或与复平面原点相切）的球面。

在复球面上，$$ \forall\theta $$，极限 $$ \lim_{\substack{\rho\to\infty\\\rho\in\mathbb{R}}}\rho\exp(\sqrt{-1}\theta) $$ 趋向同一点，故复球面可记作

$$
\overline{\mathbb{C}}\coloneqq\mathbb{C}\cup\left\{ \infty\right\} 
$$

## 复数序列

## 复数级数

# 复变函数

## 极限

## 连续性

## 函数项级数

## 多值函数

# 微分学

## 可导性

## 解析性

## 保角变换

# 积分学

## 路径积分

## Cauchy 定理

## Cauchy 积分公式

## Cauchy 型积分

## 特殊函数

# 幂级数

## Taylor 级数

## Laurent 级数

# 留数定理

## 留数

## 定理

## 定积分

## 其他应用
