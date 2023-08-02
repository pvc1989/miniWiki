---
title: 数组及基于数组的算法
---

# 数组的实现

## 长度固定的数组

## 长度可变的数组（向量）<a href id="vector"></a>

- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Resizing Arrays](https://www.coursera.org/learn/algorithms-part1/lecture/WTFO7/resizing-arrays)
- MIT
  - Text: Sect-17.4  Dynamic tables
  - Video: [6.006/Lecture 9: Table Doubling, <del>Karp-Rabin</del>](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-9-table-doubling-karp-rabin)

# 通用排序

- VisuAlgo
  - [Sorting](https://visualgo.net/en/sorting)

## 平方排序

- Princeton
  - Text: [Sect-2.1  Elementary Sorts](https://algs4.cs.princeton.edu/21elementary)
  - Video: [Part-1/Week-2  Elementary Sorts](https://www.coursera.org/learn/algorithms-part1/supplement/erHuw/lecture-slides)

### 冒泡排序

### 选择排序

### 插入排序

- MIT
  - Text: Sect-2.1  Insertion sort

## Shell 排序<a href id="shellsort"></a>

- Princeton
  - Text: *Shellsort* in [Sect-2.1  Elementary Sorts](https://algs4.cs.princeton.edu/21elementary)
  - Video: [Part-1/Week-2  Shellsort](https://www.coursera.org/learn/algorithms-part1/lecture/zPYhF/shellsort)

## 归并排序

- Princeton
  - Text: [Sect-2.2  Mergesort](https://algs4.cs.princeton.edu/22mergesort)
  - Video: [Part-1/Week-3  Mergesort](https://www.coursera.org/learn/algorithms-part1/supplement/4E9fa/lecture-slides)
- MIT
  - Text: Sect-2.3.1  The divide-and-conquer approach

### Tim 排序

- [Wikipedia](https://en.wikipedia.org/wiki/Timsort)
- Libraries
  - C++: [`std::stable_sort`](https://en.cppreference.com/w/cpp/algorithm/stable_sort) in `<algorithm>`
  - Java8: [`java.util.Arrays.sort(Object[] a)`](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#sort-java.lang.Object:A-)
  - Python:
    - [`list.sort(*, key=None, reverse=False)`](https://docs.python.org/3/library/stdtypes.html#list.sort) stably sorts the list in place.
    - [`sorted(iterable, *, key=None, reverse=False)`](https://docs.python.org/3/library/functions.html#sorted) returns a new stably sorted list from the items in `iterable`.

## 快速排序

- Princeton
  - Text: [Sect-2.3  Quicksort](https://algs4.cs.princeton.edu/23quicksort)
  - Video: [Part-1/Week-3  Quicksort](https://www.coursera.org/learn/algorithms-part1/supplement/efbDN/lecture-slides)
- MIT
  - Text: Chap-7  Quicksort
- Libraries
  - C: [`qsort`](https://en.cppreference.com/w/c/algorithm/qsort) in `<stdlib.h>`
  - C++: [`std::sort`](https://en.cppreference.com/w/cpp/algorithm/sort) in `<algorithm>`
  - Java8: [`java.util.Arrays.sort(int[] a)`](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#sort-int:A-)

## 应用

### 凸包

- VisuAlgo
  - [Convex Hull](https://visualgo.net/en/convexhull)
- Princeton
  - Video: [Part-1/Week-2  Convex Hull](https://www.coursera.org/learn/algorithms-part1/lecture/KHJ1t/convex-hull)
- MIT
  - Text: Sect-33.3  Finding the convex hull

### 共线点

- Princeton
  - Text: [Sect-2.5  Sorting Applications](https://algs4.cs.princeton.edu/25applications)
  - Programming Assignment: [Part-1/Week-3  Collinear Points](https://www.coursera.org/learn/algorithms-part1/programming/prXiW/collinear-points)

# 通用查找

## 顺序查找

## 二分查找

## 排列

```cpp
template< class BidirIt, class Compare >
constexpr bool next_permutation( BidirIt first, BidirIt last, Compare comp );
```

将范围 `[first, last)` 视为其所含元素的一种排列，若该排列在字典序（元素之间用 `comp` 比较）意义下的下一个排列存在，则将 `[first, last)` 置为新的排列，并返回 `true`；否则返回 `false`，并将 `[first, last)` 置于 `sort(first, last, comp)` 后的状态。

用法示例：

```cpp
// Given an array nums of distinct integers, return all the possible permutations.
vector<vector<int>> permute(vector<int>& nums) {
  auto ans = vector<vector<int>>();
  sort(nums.begin(), nums.end());
  do {
    ans.push_back(nums);
  } while (next_permutation(nums.begin(), nums.end()));
  return ans;
}
```

可能的实现：

```cpp
template< class BidirIt, class Compare >
bool next_permutation(BidirIt first, BidirIt last, Compare comp) {
  // 将 [first, last) 变换为等价的 [r_first, r_last)
  auto r_first = std::make_reverse_iterator(last);
  auto r_last = std::make_reverse_iterator(first);
  // 从右向左，寻找最长的单调 range，记作 [r_first, left)
  auto left = std::is_sorted_until(r_first, r_last, comp);

  if (left != r_last) {
    // [r_first, left) 中至少有 1 个元素的值“大于” *left，找到最“小”（原 range 最右）的那个
    auto right = std::upper_bound(r_first, left, *left);
    std::iter_swap(left, right);
    // 交换后，[r_first, left) 依然为单调 range，且 *left 被尽可能小地变“大”了
  }

  std::reverse(r_first, left);  // 反转后，[left.base(), last) 单调递增
  return left != r_last;
}
```

# 并查集

- VisuAlgo
  - [Union--Find DS](https://visualgo.net/en/ufds)
- Princeton
  - Text: [Sect-1.5  Union--Find](https://algs4.cs.princeton.edu/15uf/)
  - Video: [Part-1/Week-1  Union--Find](https://www.coursera.org/learn/algorithms-part1/supplement/bcelg/lecture-slides)
  - Programming Assignment: [Part-1/Week-1  Percolation](https://www.coursera.org/learn/algorithms-part1/programming/Lhp5z/percolation)

LeetCode
- [1202. Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/)
- [1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)
