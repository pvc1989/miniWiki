---
title: 常用策略
---

# 分而治之

# 动态规划

## 算法框架

### 等价描述

『动态规划 (Dynamic Programming, DP)』
- ≈ 小心的暴力枚举
  - DP 通常具有多项式复杂度。
  - 一般的暴力枚举通常具有指数复杂度。
- ≈『猜测 (guessing)』+『递归 (recursion)』+『备忘 (memoization)』
  - 若按子问题的拓扑顺序，则可（自动）改写为『自底向上 (bottom-up)』的非递归形式。

### 基本步骤

1. 定义子问题（关键步）。
1. 猜测搜索方向（通常直接枚举可用路径）。
1. 利用子问题的解（获得当前子问题的局部最优解）。
1. 递归 + 备忘（或『自底向上』建表，以避免递归）。
1. 返回原问题的解（直接返回最后一个子问题的解，或组合某几个子问题的解）。

## 应用

### 最短路径

[Bellman–Ford](./graph.md#Bellman–Ford)

### 段落分行

在某些单词前换行，使整段文字的 `badness` 值最小。

```python
# h := head, l := length, s := size
dp[h] = min(badness(h, l) + dp[h + l] for length in range(1, s - h))
```

### 矩阵乘法

```python
# h := head, t := tail, l := length
dp[h][t] = min(dp[h][h+l] + cost(h+l, h+l+1) + dp[h+l+1][t] for l in range(1, t-l-2))
```

### 编辑距离

- 原问题：将 `x[0:m)` 变为 `y[0:n)` 的最小代价。
- 子问题：将 `x[i:m)` 变为 `y[j:n)` 的最小代价。

```cpp
dp[i][j] = min(
  dp[i+1][j  ] + min(del(x[i]), add(x[i])),
  dp[i  ][j+1] + min(del(y[j]), add(y[j])),
  dp[i+1][j+1] + min(del(x[i], y[j]), replace(x[i], y[j]))
);
```

[LeetCode-72](./leetcode/72.edit-distance.md)

#### 最长公共子列

[LeetCode-1143](https://leetcode.com/problems/longest-common-subsequence/)

### 0-1 背包

给定 `n` 种货物（第 `i` 种货物的大小为 `(Int) size[i]`，价值为 `(real) value[i]`），求容量为 `(Int) room` 的『背包 (knapsack)』所能装下的总价最大的货物。

```cpp
for (int i = value.size(); i >= 0; --i) {
  for (int rest = room; rest > 0; --rest) {
    dp[i][rest] = dp[i+1][rest];
    if (rest > size[i]) {
      dp[i][rest] = max(dp[i][rest], dp[i+1][rest - size[i]] + value[i]);
    }
  }
}
```

- 时间复杂度 `Θ(n * room)` 是『伪多项式的 (pseudopolynomial)』：
  - 若 `Int` 为可变字长的整数类型，则每个 `size[i]` 需要 `lg(room)`『位 (bit)』来表示，故输入规模为 `O(n * lg(room))`，此时复杂度 `Θ(n * room)` 是『指数的』。
  - 若 `Int` 为固定字长的整数类型，则 `lg(room)` 可视为常数，此时复杂度 `Θ(n * room)` 是『多项式的』。
- 空间复杂度『伪多项式的』：
  - 二维表格 `dp[][]` 需要 `Θ(n * room)` 存储空间。
  - 实际建表时，只用到两个长度为 `Θ(room)` 的一维数组。

#### 硬币找零

目标：用最少数量的硬币达到目标面值。

[LeetCode-322](https://leetcode.com/problems/coin-change)

#### 等分集合

目标：将集合划分为若干子集，使各子集元素之和相等。

[LeetCode-416](https://leetcode.com/problems/partition-equal-subset-sum/)

### 正则表达式

[LeetCode-10](./leetcode/10.regular-expression-matching.md)

# 贪心选择

## 应用

### 连续背包

### 最多相容区间

从一组端点固定的区间中，选出尽可能多的相容区间：

1. 按区间右端排序。
1. 有剩余区间待选：
   1. 选出右端最小的区间。
   1. 忽略与之不相容的区间。

[LeetCode-435](./leetcode/435.non-overlapping-intervals.md)

### 最大股票收益

#### 允许单次交易

#### 允许多次交易

#### 引入冷却机制

```cpp
int max_profit_if_sell_next = max(
    max_profit_if_sell_curr,
    max_profit_if_hold_curr + price_next);
int max_profit_if_hold_next = max(
    max_profit_if_hold_curr,
    max_profit_if_sell_prev - price_next);
```

[LeetCode-309](./leetcode/309.best-time-to-buy-and-sell-stock-with-cooldown.md)
