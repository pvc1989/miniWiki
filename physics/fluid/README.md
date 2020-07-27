---
title: 流体力学
---

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

