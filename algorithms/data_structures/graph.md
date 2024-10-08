---
title: 图 (Graphs)
---

# 图的实现

- VisuAlgo: [Graph Structures](https://visualgo.net/en/graphds) and [Graph Traversal](https://visualgo.net/en/dfsbfs)
- Princeton
  - Text: [Sect-4.1  Undirected Graphs](https://algs4.cs.princeton.edu/41graph) and [Sect-4.2  Directed Graphs](https://algs4.cs.princeton.edu/42digraph)
  - Video: [Part-2/Week-1  Undirected Graphs](https://www.coursera.org/learn/algorithms-part2/supplement/NlsQF/lecture-slides) and [Part-2/Week-1  Directed Graphs](https://www.coursera.org/learn/algorithms-part2/supplement/qRjk3/lecture-slides)
- MIT
  - Text: Sect-B.4  Graphs

## 邻接矩阵

## 邻接链表

```cpp
template <class Vertex>
struct StaticGraph {
  List<List<Vertex>> neighbors;
};
```

## 邻接集合

```cpp
template <class Vertex>
struct DynamicGraph {
  Map<Vertex, Set<Vertex>> neighbors;
};
```

## 隐式表示

```cpp
template <class Vertex>
struct ImplicitGraph {
  List<Vertex> const& GetNeighbors(Vertex const&);
};
```

# 广度优先搜索

## 递归实现<a href id="BFS-Recursive"></a>

## 循环实现<a href id="BFS-Iterative"></a>

```python
def breadth_first_search(graph, source):
  vertex_to_parent = {source: None}
  vertex_to_level  = {source: 0}
  current_level = 0
  this_level = [source]
  while len(this_level):
    next_level = []
    for u in this_level:
      for v in graph.get_neighbors(u):
        if v not in vertex_to_level:
          vertex_to_level[v] = current_level
          vertex_to_parent[v] = u
          next_level.add(v)
    this_level = next_level
    current_level += 1
  return vertex_to_parent, vertex_to_level
```

Complexity:
- $Θ(V+E)$ time
- $Θ(V+E)$ space

## 应用<a href id="BFS-应用"></a>

### 最短祖先路径

- Princeton
  - Programming Assignment: [WordNet](https://www.coursera.org/learn/algorithms-part2/programming/BCNsp/wordnet)

![](https://coursera.cs.princeton.edu/algs4/assignments/wordnet/wordnet-sca.png)

```python
def find_shortest_ancestral_path(graph, u, v):
  u_tree = build_bfs_tree(graph, u) # Θ(V + E)
  v_tree = build_bfs_tree(graph, v) # Θ(V + E)
  d_min = int('inf')
  ancestor = None
  for x in u_tree: # at most V steps
    if x not in v_tree: # Θ(1)
      continue
    d = u_tree.get_depth(x) + v_tree.get_depth(x): # Θ(1)
    if d_min > d: # Θ(1)
      d_min = d
      ancestor = x
  return d_min, ancestor
```

Complexity:
- $Θ(V+E)$ time
- $Θ(V+E)$ space

### 跳跃游戏

允许从 `i` 跳到合法的 `i + 1`、`i - 1` 及 `a[k] == a[i]` 的 `k`，
求：从 `0` 到 `n - 1` 至少需要跳几次？

[LeetCode-1345](./leetcode/1345.jump-game-iv.cpp)

# 深度优先搜索

## 递归实现<a href id="DFS-Recursive"></a>

```python
def depth_first_search(graph):
  vertex_to_parent = dict()
  for source in graph.vertices:
    if source not in vertex_to_parent:
  		_depth_first_visit(graph, source, vertex_to_parent)
  return vertex_to_parent

def _depth_first_visit(graph, source, vertex_to_parent):
  for v in graph.get_neighbors(source):
    if v not in vertex_to_parent:
      vertex_to_parent[v] = source
      _depth_first_visit(graph, v, vertex_to_parent)
```

Complexity:
- $Θ(V+E)$ time
- $Θ(V+E)$ space

## 应用<a href id="DFS-应用"></a>

### 环路检测

Edge Classification:
- tree edge: to child
- forward edge: to descendant (only in digraph)
- back edge: to ancestor
- cross edge: to another subtree (only in digraph)

**Theorem.** Graph has a cycle $\iff$ DFS has a back edge.

DAG: Directed Acylic Graph.

### 拓扑排序<a href id="topo-sort"></a>

- Idea: sort vertices by the reverse of DFS finishing times, i.e. time at which `_depth_first_visit()` finishes.
- Libraries
  - Python3: [`graphlib.TopologicalSorter`](https://docs.python.org/3/library/graphlib.html#graphlib.TopologicalSorter) provides functionality to topologically sort a graph of hashable nodes.
- Applications:
  - Job scheduling.

### 数独求解器

[LeetCode-37](./leetcode/37.sudoku-solver.cpp)

### 跳跃游戏

允许从 `i` 跳到合法的 `i + a[i]` 或 `i - a[i]`，
求：是否能从 `0` 跳到某个 `a[k] == 0` 的 `k`？

[LeetCode-1306](./leetcode/1306.jump-game-iii.cpp)

# 最小展开树

- VisuAlgo
  - [Min Spanning Tree](https://visualgo.net/en/mst)
- Princeton
  - Text: [Sect-4.3  Minimum Spanning Trees](https://algs4.cs.princeton.edu/43mst)
  - Video: [Part-2/Week-2  Minimun Spanning Trees](https://www.coursera.org/learn/algorithms-part2/supplement/tda2O/lecture-slides)

## 贪心策略

- $W\colon E\to\mathbb{R}$ or $W\colon V\times V\to\mathbb{R}$ for an edge.
- $W(v_0,\dots,v_{k})\coloneqq\sum_{i=0}^{k-1}W(v_i,v_{i+1})$ for a path.

If negative weight edges are present, the algorithm should find negative weight cycles.

**Problem.** Given an undirected graph $G = (V,E)$ and edge weights $W\colon E\to \mathbb{R}$, find a spanning tree $T$ that minimizes $\sum_{e\in T}W(e)$.

**Definition.** The *contraction* of an edge $e\coloneqq\{u,v\}$ in a graph $G$ is to merge the vertices connected by $e$ and create a new vertex. The new graph is denoted as $G/e$.

**Lemma (Optimal Substructure).** Suppose $e\coloneqq\{u,v\}$ is an edge of some MST of $G$. If $T'$ is an MST of $G/e$, then $T'\cup\{e\}$ is an MST of $G$.

**Lemma (Greedy Choice).** For any cut $(S,V\setminus S)$ in a weighted graph $G=(V,E,W)$, any least-weight crossing edge $e\coloneqq\{u\in S,v\in V\setminus S\}$ is in some MST of $G$.

## Kruskal

- Idea:
  - Maintain connected components by a [Union–Find data structure](./array.md#并查集).
  - Greedily choose the globally lowest-weight edge that connect two components.
- Complexity:
  - $Θ(V)$ for building the `UnionFind` data structure of vertices.
  - $Θ(E)$ for `sort()` if $W$ is `int`-valued and using [Radix Sort](./string.md#基数排序).
  - $Θ(E)$ calls of `UnionFind.connected()` and `UnionFind.union()`, which can be amortized $Θ(\alpha(V))$.

```python
def GetMinSpanTreeByKruskal(vertices, edges, get_weight):
  # Initialization:
  mst = set() # edges
  uf = UnionFind(vertices) # one component for each vertex
  sort(edges, get_weight) # may be linear
  # Greedily choose the lowest-weight edge:
  for (u, v) in edges:
    if not uf.connected(u, v):
      uf.union(u, v)
      mst.add((u, v))
  # Termination:
  return mst
```

LeetCode
- [1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)

## Prim

- Idea (like [Dijkstra's algorithm for Shortest Path](#Dijkstra)):
  - Maintain a `MinPQ` on $V\setminus S$, where $d(S, v)\coloneqq\min_{u\in S}\{W(u, v)\}$ is used as $v$'s `key`.
  - Greedily choose the closest vertex from set $V\setminus S$ and add it to set $S$.
- Complexity:
  - $Θ(V)$ calls of `MinPQ.pop_min()`
  - $Θ(E)$ calls of `MinPQ.change_key()`, which can be amortized $Θ(1)$ if using [Fibonacci Heap](./queue.md#fib-heap).
  - $Θ(V+E)$ space

```python
def GetMinSpanTreeByPrim(vertices, edges, get_weight):
  # Initialization:
  mst = dict() # vertex to parent
  pq = MinPQ() # v.key := d(S, v)
  for v in vertices:
    v.parent = None
    v.key = float('inf')
    pq.add(v)
  # Choose the root (arbitrarily):
  u = pq.pop_min()
  mst[u] = None
  for v in u.neighbors:
    v.key = get_weight(u, v) # float up in the MinPQ
    v.parent = u
  # Greedily choose the next (V-1) vertices:
  while len(pq):
    u = pq.pop_min()
    mst[u] = u.parent
    for v in u.neighbors:
      if (v not in mst) and (get_weight(u, v) < v.key):
        pq.change_key(v, key=get_weight(u, v))
        v.parent = u
  # Termination:
  return mst
```

# 单源最短路径

- VisuAlgo
  - [Single-Source Shortest Paths](https://visualgo.net/en/sssp)
- Princeton
  - Text: [Sect-4.4  Shortest Paths](https://algs4.cs.princeton.edu/44sp)
  - Video: [Part-2/Week-2  Shortest Paths](https://www.coursera.org/learn/algorithms-part2/supplement/BZTAt/lecture-slides)
  - Programming Assignment: [Seam Carving](https://www.coursera.org/learn/algorithms-part2/programming/cOdkz/seam-carving)

## 算法框架

```python
def find_shortest_path(source, graph, get_weight):
  # Initialization:
  vertex_to_length = dict()
  vertex_to_parent = dict()
  for v in graph.vertices:
    vertex_to_length[v] = float('inf')
    vertex_to_parent[v] = None
  vertex_to_length[source] = 0
  # Relaxation:
  while True:
    u, v = _select_edge(graph) # return (None, None) if
    # for all (u, v), there is d[v] <= d[u] + w(u, v).
    if u is None:
      break # go to the termination step
    d = vertex_to_length[u] + get_weight(u, v)
    if vertex_to_length[v] > d: # need relaxation
      vertex_to_length[v] = d
      vertex_to_parent[v] = u
  # Termination:
  return vertex_to_length, vertex_to_parent
```

## Dijkstra

- Assumption: non-negative edge weights.
- Idea (like [Prim's algorithm for Minimum Spanning Tree](#Prim)):
  - Maintain a set $S$ of vertices whose final shortest path weights have been determined.
  - *Greedily* choose the closest vertex from set $V\setminus S$ and add it to set $S$.
- Correctness:
  - Relaxation is safe.
  - When $u$ is added to $S$, there is $d(u) = \delta(u)$.
- Complexity:
  - $Θ(V)$ calls of `MinPQ.insert(Vertex, Key)`
  - $Θ(V)$ calls of `MinPQ.pop_min()`
  - $Θ(E)$ calls of `MinPQ.decrease(Vertex, Key)`, which can be amortized $Θ(1)$ if using [Fibonacci Heap](./queue.md#fib-heap).

```python
def find_shortest_path(source, graph, get_weight):
  # Initialization:
  unfinished_vertices = MinPQ()
  vertex_to_length = dict()
  vertex_to_parent = dict()
  for v in graph.vertices:
    unfinished_vertices.insert(v, float('inf'))
    vertex_to_length[v] = float('inf')
    vertex_to_parent[v] = None
  unfinished_vertices.decrease(source, 0)
  vertex_to_length[source] = 0
  # Relaxation:
  while len(unfinished_vertices):
    u = unfinished_vertices.pop_min()
    for v in u.neighbors:
      d = vertex_to_length[u] + get_weight(u, v)
      if vertex_to_length[v] > d: # need relaxation
        unfinished_vertices.decrease(v, d)
        vertex_to_length[v] = d
        vertex_to_parent[v] = u
  # Termination:
  return vertex_to_length, vertex_to_parent
```

In practice, the `decrease()` operation may be replaced by a `insert()` operation, which inserts a new copy of the `Vertex` with a decreased `Key`.
The deletion of the old copy is delayed or even skipped.

```python
# no-decreasing version
def find_shortest_path(source, graph, get_weight):
  # Initialization:
  unfinished_vertices = MinPQ()
  finished_vertices = set()
  vertex_to_length = dict()
  vertex_to_parent = dict()
  for v in graph.vertices:
    unfinished_vertices.insert(v, float('inf'))
    vertex_to_length[v] = float('inf')
    vertex_to_parent[v] = None
  unfinished_vertices.insert(source, 0)
  vertex_to_length[source] = 0
  # Relaxation:
  while len(finished_vertices) < len(graph.vertices):
    u = unfinished_vertices.pop_min()
    if u not in finished_vertices:
      for v in u.neighbors:
        d = vertex_to_length[u] + get_weight(u, v)
        if vertex_to_length[v] > d: # need relaxation
          unfinished_vertices.insert(v, d)
          vertex_to_length[v] = d
          vertex_to_parent[v] = u
      finished_vertices.insert(u)
  # Termination:
  return vertex_to_length, vertex_to_parent
```

LeetCode
- [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)

## Bellman–Ford

- MIT:
  - Video: [6.006/Lecture 17: Bellmen–Ford](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-17-bellman-ford)
- Assumption:
  - Allow negative edge weights.
  - Report cycles with negetive weights.
- Complexity:
  - For general graphs, $Θ(VE)$ calls of `relax()`:
    ```python
    def find_shortest_path(source, graph, get_weight):
      for i in range(len(graph.vertices) - 1):
        for u, v in graph.edges:
          relax(u, v, get_weight(u, v))
      # One more pass to find negative cycles:
      for u, v in graph.edges:
        if relaxable(u, v, get_weight(u, v)):
          raise Exception("There exists a negative cycle!")
    ```
  - For DAGs, $Θ(V+E)$ for [topological sort](#topo-sort) and $Θ(E)$ calls of `relax()`:
    ```python
    def find_shortest_path(source, graph, get_weight):
      sorted_vertices = topological_sort(graph)
      for u in sorted_vertices:
        for v in graph.get_neighbors(u):
          relax(u, v, get_weight(u, v))
    ```

LeetCode
- [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

# 全源最短路径

对于只含非负边的图，可以对每个点调用 [Dijkstra](#dijkstra)。

以下算法适用于含负边、但不含负环的图。

## 矩阵乘幂类比

令 $$l_{ij}^{(r)}$$ 表示 $$i\to j$$ 至多含 $$r$$ 条边的最短路径，则初始值

$$
l_{ij}^{(0)}=
\begin{cases}
0,&i=j,\\
\infty,&i=j,\\
\end{cases}
$$

递归地有

$$
l_{ij}^{(r)}=\min_{k=1}^{V}\qty(l_{ik}^{(r-1)}+w_{kj}),
$$

$$\forall r$$ 有 $$V\times V$$ 个值需要更新，故整体时间复杂度为 $$\order{V^4}$$.

类比矩阵乘法，有如下对应关系：

|                  矩阵乘法                   |                          最段路径                          |
| :-----------------------------------------: | :--------------------------------------------------------: |
| $$c_{ij}=\sum_{k=1}^{V}a_{ik}\cdot b_{kj}$$ | $$l_{ij}^{(r)}=\min_{k=1}^{V}\qty(l_{ik}^{(r-1)}+w_{kj})$$ |
|                    $$a$$                    |                       $$l^{(r-1)}$$                        |
|                    $$b$$                    |                           $$w$$                            |
|                    $$c$$                    |                        $$l^{(r)}$$                         |
|                  $$\sum$$                   |                          $$\min$$                          |
|                  $$\cdot$$                  |                           $$+$$                            |

故最短路径问题归结为矩阵乘幂：

$$
L^{(r)}=L^{(r-1)}\cdot W=\cdots=W^{r},
$$

利用[分治](./strategy.md#分而治之)策略，时间复杂度可降为 $$\order{V^3\lg V}$$.

## Floyd–Warshall

令 $$d_{ij}^{(k)}$$ 表示中间节点只含 $$1,\dots,k$$ 的最短路径，则初值及递归式分别为

$$
d_{ij}^{(k)}=
\begin{cases}
w_{ij},&k=0\\
\min\qty(d_{ij}^{(k-1)},d_{ik}^{(k-1)}+d_{kj}^{(k-1)}),&k>0,\\
\end{cases}
$$

$$\forall k$$ 有 $$V\times V$$ 个值需要更新，故整体时间复杂度为 $$\order{V^3}$$.

## Johnson

对于稀疏图，可以先用 [Bellman--Ford](#bellmanford) 改造为只含非负边的图，再用对每个 vertex 调用 [Dijkstra](#dijkstra)，故整体时间复杂度为 $$\order{V^2\lg V+VE}$$.

图的改造方法如下：

在 $$G$$ 外引入一点 $$s$$，令 $$\forall v\in V : w(s,v)=0$$，用 [Bellman--Ford](#bellmanford) 得到 $$\delta(s,v)$$，记作 $$h(v)$$。

由最短路径的三角不等式性质可以得到：

$$
w(u,v)+h(u)-h(v)\ge0\impliedby\delta(s,v)\le\delta(s,u)+w(u,v).
$$

|      $$\hat{G}$$      |            $$(\hat{V},\hat{E})$$            |
| :-------------------: | :-----------------------------------------: |
|      $$\hat{V}$$      |                $$V\cup{s}$$                 |
|      $$\hat{E}$$      |          $$E\cup\{(s,v):v\in V\}$$          |
|   $$\hat{w}(u,v)$$    |            $$w(u,v)+h(u)-h(v)$$             |
| $$\hat{\delta}(u,v)$$ | $$\hat{\delta}(u,v)=\delta(u,v)+h(u)-h(v)$$ |

# 最大流、最小割

- VisuAlgo
  - [Network Flow](https://visualgo.net/en/maxflow)
- Princeton
  - Text: [Sect-6.4  Maxflow](https://algs4.cs.princeton.edu/64maxflow)
  - Video: [Part-2/Week-3  Maximum Flow and Minimum Cut](https://www.coursera.org/learn/algorithms-part2/supplement/qKIDx/lecture-slides)
  - Programming Assignment: [Baseball Elimination](https://www.coursera.org/learn/algorithms-part2/programming/hmYRI/baseball-elimination)

## Ford–Fulkerson

## 最大流–最小割定理
