---
title: 常用策略
---

# 分而治之

## [主定理](./complexity.md#master-thm)

$$
T(N) =a\cdot T(N/b)+f(N)=\mathopen{\Theta}\left(N^{\log_{b}a}\right)+\sum_{\nu=0}^{n-1}a^{\nu}\cdot f(N/b^{\nu}),\qquad n\coloneqq\log_{b}N
$$

# 动态规划

## 框架

### 等价描述

**动态规划 (Dynamic Programming, DP)**

- ≈ 省力的暴力枚举
  - 简单的暴力枚举，通常具有指数复杂度。
  - 利用动态规划，可降到（伪）多项式复杂度。
- ≈ **递归 (recursion)** + **备忘 (memoization)**
  - **自顶向下 (top-down)** 的递归代码，会因递归函数调用消耗一定的计算资源，但
    - 是描述**子问题 (subproblem)** 的最佳方式，通常比相应的非递归版本直观。
    - 子问题的依赖关系隐含在递归调用中，无需刻意维护。
    - 不被原问题所依赖的子问题，不会被求解，可节约计算资源。
  - **自底向上 (bottom-up)** 的非递归代码，通常具有更高的运行效率，但
    - 通常不如相应的递归版本直观。
    - 需按子问题的**拓扑顺序 (topological order)** 逐层求解。
    - 不被原问题所依赖的子问题，也会被求解，会浪费计算资源。
- ≈ **有向无环图 (Directed Acyclic Graph, DAG)** 上的最短路径搜索
  - 以子问题为结点
  - 以依赖关系为邻边
  - 以优化目标为路径开销

### 基本步骤

1. 定义子问题（关键步）。
1. 猜测搜索方向（通常直接枚举可用路径）。
1. 利用子问题的解（获得当前子问题的局部最优解）。
1. 递归 + 备忘（或*自底向上*建表，以避免递归）。
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

LeetCode-72

- 【[描述](https://leetcode.com/problems/edit-distance/description/)】给定两个字符串 (`x `, `y`)，求使得二者相同的最少操作 (`add`, `del`, `replace`) 次数。
- 【[解答](./leetcode/72.edit-distance.cpp)】子问题：将 `x[i:m)` 变为 `y[j:n)` 的最小代价。

```cpp
dp[i][j] = min(
  dp[i+1][j  ] + min(del(x[i]), add(x[i])),
  dp[i  ][j+1] + min(del(y[j]), add(y[j])),
  dp[i+1][j+1] + min(del(x[i], y[j]), replace(x[i], y[j]))
);
```

#### 最长公共子列

LeetCode-1143

- 【[描述](https://leetcode.com/problems/longest-common-subsequence/description/)】给定两个字符串 (`x `, `y`)，求最长公共子列（只允许 `del` 操作）。
- 【[解答](./leetcode/1143.longest-common-subsequence.cpp)】

### 0-1 背包

- 【描述】给定 `n_items` 种货物（第 `i` 种货物的大小为 `(Int) size[i]`，价值为 `(Real) value[i]`），求容量为 `(Int) max_room` 的**背包 (knapsack)** 所能装下的总价最大的货物。
- 【解答】
  ```cpp
  // `dp[i][r]` := max value of items in range `[i, n)` with `r` remaining room
  for (int item = value.size() - 1; 0 <= item; --item) {
    for (int room = max_room; 0 < room; --room) {
      dp[item][room] = dp[item + 1][room];
      if (size[item] < room) {
        auto new_value = dp[item + 1][room - size[item]] + value[item];
        dp[item][room] = max(dp[item][room], new_value);
      }
    }
  }
  return dp[0][max_room];
  ```
- 时间复杂度 `Θ(n_items * max_room)` 是**伪多项式的 (pseudopolynomial)**：
  - 若 `Int` 为可变字长的整数类型，则每个 `size[i]` 需要 `lg(max_room)` bits 来表示，故输入规模为 `O(n_items * lg(max_room))`，此时复杂度 `Θ(n_items * maxroom)` 是*指数的*。
  - 若 `Int` 为固定字长的整数类型，则 `lg(room)` 可视为常数，此时复杂度 `Θ(n_items * max_room)` 是*多项式的*。
- 空间复杂度是*伪多项式的*：
  - 二维表格 `dp[][]` 需要 `Θ(n_items * max_room)` 存储空间。
  - 实际建表时，只用到两个长度为 `Θ(max_room)` 的一维数组。

#### 硬币找零

LeetCode-322

- 【[描述](https://leetcode.com/problems/coin-change/description/)】给定各面值数量不限的硬币，求达到目标面值的最少硬币数量。
- 【[解答](./leetcode/322.coin-change.cpp)】

#### 二分集合

LeetCode-416

- 【[描述](https://leetcode.com/problems/partition-equal-subset-sum/description/)】将只含正数的集合划分为两个子集，使二者的元素之和相等。
- 【[解答](./leetcode/416.partition-equal-subset-sum.cpp)】

### 旅行商问题

【**旅行商问题 (Travelling Salesman Problem, TSP)**】给定 $V$ 座城市及 $E$ 条（连接其中两市的）道路的长度，求一条到访每座城市各一次且总长度最短的回路。

#### 最短超字符串

LeetCode-943

- 【[描述](https://leetcode.com/problems/find-the-shortest-superstring/)】给定 $N$ 个最长为 $W$ 字符且两两互不覆盖的字符串，求一个能覆盖所有词的**最短超字符串 (shortest superstring)**。
- 【[解答](./leetcode/943.find-the-shortest-superstring.cpp)】令 `dp(suffix_set, i)` 返回由子集 `suffix_set` 中的字符串所构成、且以第 `i` 个字符串为首的后缀的最大重叠字符个数。
  - `dp(suffix_set+(1<<i), i) = max_{j ∈ suffix_set}(overlap(s[i], s[j]) + dp(suffix_set, j))`
  - 复杂度：时间 $Θ(N^2\cdot(2^N+W))$，空间 $Θ(N\cdot(2^N+W))$

### 正则表达式

#### `*` 表示重复前一字符

LeetCode-10

- 【[描述](https://leetcode.com/problems/regular-expression-matching/description/)】`.` 表示任意字符、`*` 表示重复前一字符
- 【[解答](./leetcode/10.regular-expression-matching.cpp)】

#### `*` 表示任意字符子串

LeetCode-44

- 【[描述](https://leetcode.com/problems/wildcard-matching/description/)】`?` 表示任意字符、`*` 表示任意字符子串
- 【[解答](./leetcode/44.wildcard-matching.cpp)】

### 最大股票收益

【**交易 (transaction)**】某支股票，在第 `i` 天被买入，在第 `j` 天被卖出，称作一次交易。

#### 至多 $2$ 次交易

LeetCode-123

- 【[描述](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/)】给定一段时间内的价格，至多 $2$ 次交易且互不重叠，求最大收益。
- 【[解答](./leetcode/123.best-time-to-buy-and-sell-stock-iii.cpp)】有复杂度为 $O(2n^2)$​ 的 DP 解法
  $$
  P_{[0, n)}^{2} = \max_{i\in[0,n)} \left(P_{[0,i)}^{1} + P_{[i,n)}^{1}\right)
  $$
  其中 $P_{[i,j)}^{t}$ 表示*最早于第 $i$ 天买入、最晚于第 $(j-1)$ 天卖出、至多完成 $t$ 次交易的最大收益*。

#### 至多 $k$ 次交易

LeetCode-188

- 【[描述](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/)】给定一段时间内的价格，至多 $k$ 次交易且互不重叠，求最大收益。
- 【[解答](./leetcode/188.best-time-to-buy-and-sell-stock-iv.cpp)】仿照上题，有复杂度为 $O(kn^2)$ 的 DP 解法（另有复杂度为 $O(kn)$ 的[贪心](#k-transactions)解法）：
  $$
  P_{[0, j)}^{t} = \max_{i\in[0,j)} \left(P_{[0,i)}^{t-1} + P_{[i,j)}^{1}\right)\qquad j\in[0,n)
  $$
  其中 $P_{[i,j)}^{t}$ 表示*最早于第 $i$ 天买入、最晚于第 $(j-1)$ 天卖出、至多完成 $t$ 次交易的最大收益*。

### 二叉树覆盖

[LeetCode-968](./leetcode/968.binary-tree-cameras.cpp)

# 贪心选择

## 框架

## 应用

### 连续背包

### 任务调度

#### 限制任务间隔

[LeetCode-621](./leetcode/621.task-scheduler.cpp)

#### 最多相容区间

[LeetCode-435](./leetcode/435.non-overlapping-intervals.cpp)

从一组端点固定的区间中，选出尽可能多的相容区间：

1. 按区间右端排序。
1. 有剩余区间待选：
   1. 选出右端最小的区间。
   1. 忽略与之不相容的区间。

### 重构队列

`h` 由低到高、`k` 由大到小，在剩余空位中顺序查找相应位置。

[LeetCode-406](./leetcode/406.queue-reconstruction-by-height.cpp)

### 糖果分发

[LeetCode-135](./leetcode/135.candy.cpp)

### 最大性能

[LeetCode-1383](./leetcode/1383.maximum-performance-of-a-team.cpp)

### 最大股票收益

#### 至多一次交易

[LeetCode-121](./leetcode/121.best-time-to-buy-and-sell-stock.cpp)

#### 不限交易次数

[LeetCode-122](./leetcode/122.best-time-to-buy-and-sell-stock-ii.cpp)

#### 引入冷却机制

[LeetCode-309](./leetcode/309.best-time-to-buy-and-sell-stock-with-cooldown.cpp)

$$
H_{d+1} = \mathopen{\max}\left(H_{d}, S_{\boxed{d-1}}-P_{d+1}\right)
\qquad
S_{d+1} = \mathopen{\max}\left(S_{d}, H_{\boxed{d}}+P_{d+1}\right)
$$

#### 引入交易成本

[LeetCode-714](./leetcode/714.best-time-to-buy-and-sell-stock-with-transaction-fee.cpp)

$$
H_{d} = \mathopen{\max}\left(H_{d-1}, S_{d-1}-P_d\right)
\qquad
S_{d} = \mathopen{\max}\left(S_{d-1}, H_{d-1}+P_d-F\right)
$$

#### 至多 $k$ 次交易<a href id="k-transactions"></a>

[LeetCode-188](./leetcode/188.best-time-to-buy-and-sell-stock-iv.cpp)

$$
H_{d}^{t} = \mathopen{\max}\left(H_{d-1}^{t-1}, S_{d-1}^{\boxed{t-1}}-P_d\right)
\qquad
S_{d}^{t} = \mathopen{\max}\left(S_{d-1}^{t-1}, H_{d-1}^{\boxed{t}}+P_d\right)
$$

其中
- $P_d$ 表示*第 $d$ 天的股价*。
- $S_{d}^{t}$ 表示*第 $d$ 天为卖出状态、前 $d$ 天至多卖出 $t$ 次的最大收益*。
- $H_{d}^{t}$ 表示*第 $d$ 天为持有状态、前 $d$ 天至多买入 $t$ 次的最大收益*。

