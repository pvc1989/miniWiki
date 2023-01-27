---
title: 数据结构
---

# 资源

## 课程

### Princeton

- Textbooks:
  - [Algorithms, 4th Edition](https://algs4.cs.princeton.edu/home/)
  - [An Introduction to the Analysis of Algorithms, 2nd Edition](https://aofa.cs.princeton.edu/home/)
- Courses:
  - [Algorithms, Part 1](https://www.coursera.org/learn/algorithms-part1)
  - [Algorithms, Part 2](https://www.coursera.org/learn/algorithms-part2)
  - [Analysis of Algorithms](https://www.coursera.org/learn/analysis-of-algorithms)
- Libraries:
  - [Java Algorithms and Clients](https://algs4.cs.princeton.edu/code/)
  - [kevin-wayne/algs4](https://github.com/kevin-wayne/algs4) on GitHub

### MIT

- Textbook:
  - [Introduction to Algorithms, Third Edition](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)
- Courses:
  - [Introduction to Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/) taught by *Prof. Erik Demaine* and *Prof. Srini Devadas* in Fall 2011.
  - [Design and Analysis of Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/) taught by *Prof. Erik Demaine* and *Prof. Srini Devadas* and *Prof. Nancy Lynch* in Spring 2015.

## 网站

### 在线演示

- [VisuAlgo](https://visualgo.net/en): visualising data structures and algorithms through animation.

### 在线练习

- [LeetCode](https://leetcode.com/): the world's leading online programming learning platform.
  - 建议用 [leetcode.vscode-leetcode](https://marketplace.visualstudio.com/items?itemName=LeetCode.vscode-leetcode) 在本地刷题及备份。本地题库无法更新时，可先执行 `rm ~/.lc/leetcode/cache/problems.json`，再点击刷新按钮。
- [PAT](https://www.patest.cn/) (Programming Ability Test) and [PTA](https://pintia.cn) (Programming Teaching Assistant).
- [牛客网](https://www.nowcoder.com)

# [数组及基于数组的算法](./array.md)
## [数组的实现](./array.md#数组的实现)
[长度固定的数组](./array.md#长度固定的数组)、[长度可变的数组（向量）](./array.md#vector)

## [通用排序](./array.md#通用排序)
[平方排序](./array.md#平方排序)、[Shell 排序](./array.md#shellsort)、[归并排序](./array.md#归并排序)、[快速排序](./array.md#快速排序)
## [通用查找](./array.md#通用查找)
[顺序查找](./array.md#顺序查找)、[二分查找](./array.md#二分查找)
## [并查集](./array.md#并查集)

# [队列 (Queues)](./queue.md)
## [线性队列](./queue.md#线性队列)
[链表](./queue.md#链表)、[队列](./queue.md#先进先出队列)、[栈](./queue.md#后进先出队列（栈）)、[双向队列](./queue.md#双向队列)、[随机队列](./queue.md#随机队列)
## [优先队列](./queue.md#优先队列)
[二叉堆](./queue.md#二叉堆)、[Fibonacci 堆](./queue.md#fib-heap)、[van Emde Boas 树](./queue.md#vEB-tree)

# [散列 (Hashing)](./hash.md)
## [散列函数](./hash.md#散列函数)
[必要条件](./hash.md#必要条件)、[基于余数的散列函数](./hash.md#基于余数的散列函数)、[基于乘法的散列函数](./hash.md#基于乘法的散列函数)、[程序实现](./hash.md#程序实现)
## [冲突化解](./hash.md#冲突化解)
[分离链法](./hash.md#分离链法)、[开地址法](./hash.md#开地址法)
## [全域散列](./hash.md#全域散列)
[全域散列函数族](./hash.md#全域散列函数族)、[基于点乘的全域散列](./hash.md#基于点乘的全域散列)、[基于余数的全域散列](./hash.md#基于余数的全域散列)
## [完美散列](./hash.md#完美散列)
[静态字典](./hash.md#静态字典)、[二重散列](./hash.md#二重散列)

# [树 (Trees)](./tree.md)
## [树的实现](./tree.md#树的实现)
## [二叉搜索树](./tree.md#二叉搜索树)
## [平衡搜索树](./tree.md#平衡搜索树)
[AVL 树](./tree.md#AVL)、[红黑树](./tree.md#红黑树)、[B-树](./tree.md#B-树)
## [几何搜索树](./tree.md#几何搜索树)
[Kd-树](./tree.md#Kd-树)、[区间树](./tree.md#区间树)

# [图 (Graphs)](./graph.md)
## [图的实现](./graph.md#图的实现)
[邻接矩阵](./graph.md#邻接矩阵)、[邻接链表](./graph.md#邻接链表)、[邻接集合](./graph.md#邻接集合)、[隐式表示](./graph.md#隐式表示)
## [广度优先搜索](./graph.md#广度优先搜索)
[递归实现](./graph.md#BFS-Recursive)、[循环实现](./graph.md#BFS-Iterative)、[应用](./graph.md#BFS-应用)
## [深度优先搜索](./graph.md#深度优先搜索)
[递归实现](./graph.md#DFS-Recursive)、[邻接集合](./graph.md#邻接集合)、[应用](./graph.md#DFS-应用)（[环路检测](./graph.md#环路检测)、[拓扑排序](./graph.md#拓扑排序)、[数独求解器](./graph.md#数独求解器)）
## [最小展开树](./graph.md#最小展开树)
[Kruskal](./graph.md#Kruskal)、[Prim](./graph.md#Prim)
## [最短路径](./graph.md#最短路径)
[Dijkstra](./graph.md#Dijkstra)、[Bellman–Ford](./graph.md#bellmanford)
## [最大流、最小割](./graph.md#最大流、最小割)
[Ford–Fulkerson](./graph.md#fordfulkerson)、[最大流–最小割定理](./graph.md#最大流–最小割定理)

# [串 (Strings)](./string.md)
## [串的实现](./string.md#串的实现)
## [基数排序](./string.md#基数排序)
[始于末位的基数排序](./string.md#始于末位的基数排序)、[始于首位的基数排序](./string.md#始于首位的基数排序)、[三路基数快速排序](./string.md#三路基数快速排序)
## [后缀数组](./string.md#后缀数组)
## [字典树](./string.md#字典树)
[R-路字典树](./string.md#R-路字典树)、[三元搜索字典树](./string.md#三元搜索字典树)
## [子串搜索](./string.md#子串搜索)
[Knuth–Morris–Pratt](./string.md#knuthmorrispratt)、[Boyer–Moore](./string.md#boyermoore)、[Rabin–Karp](./string.md#rabinkarp)
## [正则表达式](./string.md#正则表达式)
[匹配算法](./string.md#匹配算法)（[NFA](./string.md#NFA)、[动态规划](./string.md#re-dp)）
## [数据压缩](./string.md#数据压缩)

# [复杂度](./complexity.md)
## [数学（理论）分析](./complexity.md#mathematic)
## [科学（实验）测量](./complexity.md#scientific)
## [NP-完备性](./complexity.md#NPC)

# [常用策略](./strategy.md)
## [分而治之](./strategy.md#分而治之)
## [动态规划](./strategy.md#动态规划)
## [贪心选择](./strategy.md#贪心选择)
