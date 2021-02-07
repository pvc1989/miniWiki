---
title: 复杂度 (Complexity)
---

# 数学（理论）分析<a href id="mathematic"></a>

- MIT
  - Text: Chap-3  Growth of Functions

## 递归分析

- MIT
  - Text: Chap-4  Divide-and-Conquer

### 主定理

$$
\begin{aligned}T(N) & =a\cdot T(N/b)+f(N)=\mathopen{\Theta}\left(N^{\log_{b}a}\right)+\sum_{\nu=0}^{n-1}a^{\nu}\cdot f(N/b^{\nu}),\qquad n\coloneqq\log_{b}N\\
 & =\begin{cases}
\mathopen{\Theta}\left(N^{\log_{b}a}\right) & f(N)=\mathopen{O}\left(N^{\log_{b}a}\div N^{\epsilon}\right)\\
\mathopen{\Theta}\left(N^{\log_{b}a}\cdot(\lg N)^{m+1}\right) & f(N)=\mathopen{\Theta}\left(N^{\log_{b}a}\cdot(\lg N)^{m}\right)\\
\mathopen{\Theta}\left(f(N)\right) & f(N)=\mathopen{\Omega}\left(N^{\log_{b}a}\times N^{\epsilon}\right)\land\exists c\in(0,1)\colon\ a\cdot f(N/b)\le c\cdot f(N)
\end{cases}
\end{aligned}
$$

## 概率分析

- MIT
  - Text: Chap-5  Probabilistic Analysis and Randomized Algorithms

## 摊还分析

- MIT
  - Text: Chap-17  Amortized Analysis

# 科学（实验）测量<a href id="scientific"></a>

- Princeton
  - Text: [Sect-1.4  Analysis of Algorithms](https://algs4.cs.princeton.edu/14analysis/)
  - Video: [Part-1/Week-1  Analysis of Algorithms](https://www.coursera.org/learn/algorithms-part1/supplement/mpK20/lecture-slides)

> 1. *Observe* some feature of the natural world, generally with precise measurements.
> 2. *Hypothesize* a model that is consistent with the observations.
> 3. *Predict* events using the hypothesis.
> 4. *Verify* the predictions by making further observations.
> 5. *Validate* by repeating until the hypothesis and observations agree.

# NP-完备性<a href id="NPC"></a>

- MIT
  - Text: Chap-34  NP-Completeness

## 问题规约

- Princeton
  - Text: [Sect-6.5  Reductions](https://algs4.cs.princeton.edu/65reductions)
  - Video: [Part-2/Week-6  Reductions](https://www.coursera.org/learn/algorithms-part2/supplement/OD01e/lecture-slides)

## 线性规划

- Princeton
  - Video: [Part-2/Week-6  Linear Programming](https://www.coursera.org/learn/algorithms-part2/supplement/9wPqe/lecture-slides)
- MIT
  - Text: Chap-29  Linear Programming

## 不可解性

- Princeton
  - Text: [Sect-6.6  Intractability](https://algs4.cs.princeton.edu/66intractability)
  - Video: [Part-2/Week-6  Intractability](https://www.coursera.org/learn/algorithms-part2/supplement/Nc2PX/lecture-slides)
