---
title: 常用策略
---

# 分而治之

# 动态规划

## 算法框架

## 应用

### 矩阵乘法

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

```cpp
dp[i][rest] = max(dp[i+1][rest], dp[i+1][rest - size[i]] + value[i]);
```

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
