---
title: 常用策略
---

# 分而治之

# 动态规划

## 应用

### 正则表达式

[LeetCode-10](./leetcode/10.regular-expression-matching.md)

# 贪心选择

## 应用

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
