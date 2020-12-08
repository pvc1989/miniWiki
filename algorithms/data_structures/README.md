---
title: 数据结构与算法
---

# Resources

## Courses

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

## Websites

### Study

- [VisuAlgo](https://visualgo.net/en): visualising data structures and algorithms through animation.

### Practice

- [LeetCode](https://leetcode.com/): the world's leading online programming learning platform.
- [PAT](https://www.patest.cn/): Programming Ability Test.

# Computational Complexity

## Mathematical (Theoretical) Method

- MIT
  - Text: Chap-3  Growth of Functions

### Divide-and-Conquer

- MIT
  - Text: Chap-4  Divide-and-Conquer

### Probabilitic Analysis

- MIT
  - Text: Chap-5  Probabilistic Analysis and Randomized Algorithms

### Amortized Analysis

- MIT
  - Text: Chap-17  Amortized Analysis

## Scientific (Experimental) Method

- Princeton
  - Text: [Sect-1.4  Analysis of Algorithms](https://algs4.cs.princeton.edu/14analysis/)
  - Video: [Part-1/Week-1  Analysis of Algorithms](https://www.coursera.org/learn/algorithms-part1/supplement/mpK20/lecture-slides)

> 1. *Observe* some feature of the natural world, generally with precise measurements.
> 2. *Hypothesize* a model that is consistent with the observations.
> 3. *Predict* events using the hypothesis.
> 4. *Verify* the predictions by making further observations.
> 5. *Validate* by repeating until the hypothesis and observations agree.

## NP-Completeness

- MIT
  - Text: Chap-34  NP-Completeness

### Reductions

- Princeton
  - Text: [Sect-6.5  Reductions](https://algs4.cs.princeton.edu/65reductions)
  - Video: [Part-2/Week-6  Reductions](https://www.coursera.org/learn/algorithms-part2/supplement/OD01e/lecture-slides)

### Linear Programming

- Princeton
  - Video: [Part-2/Week-6  Linear Programming](https://www.coursera.org/learn/algorithms-part2/supplement/9wPqe/lecture-slides)
- MIT
  - Text: Chap-29  Linear Programming

### Intractability

- Princeton
  - Text: [Sect-6.6  Intractability](https://algs4.cs.princeton.edu/66intractability)
  - Video: [Part-2/Week-6  Intractability](https://www.coursera.org/learn/algorithms-part2/supplement/Nc2PX/lecture-slides)

# Arrays & Vectors

## Sorting

- VisuAlgo
  - [Sorting](https://visualgo.net/en/sorting)

### Quadratic Sorts

- Princeton
  - Text: [Sect-2.1  Elementary Sorts](https://algs4.cs.princeton.edu/21elementary)
  - Video: [Part-1/Week-2  Elementary Sorts](https://www.coursera.org/learn/algorithms-part1/supplement/erHuw/lecture-slides)

#### Bubble Sort

#### Selection Sort

#### Insertion Sort

- MIT
  - Text: Sect-2.1  Insertion sort

### Shellsort

- Princeton
  - Text: *Shellsort* in [Sect-2.1  Elementary Sorts](https://algs4.cs.princeton.edu/21elementary)
  - Video: [Part-1/Week-2  Shellsort](https://www.coursera.org/learn/algorithms-part1/lecture/zPYhF/shellsort)

### Mergesort

- Princeton
  - Text: [Sect-2.2  Mergesort](https://algs4.cs.princeton.edu/22mergesort)
  - Video: [Part-1/Week-3  Mergesort](https://www.coursera.org/learn/algorithms-part1/supplement/4E9fa/lecture-slides)
- MIT
  - Text: Sect-2.3.1  The divide-and-conquer approach

#### [Timsort](https://en.wikipedia.org/wiki/Timsort)

- Libraries
  - C++: [`std::stable_sort`](https://en.cppreference.com/w/cpp/algorithm/stable_sort) in `<algorithm>`
  - Java8: [`java.util.Arrays.sort(Object[] a)`](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#sort-java.lang.Object:A-)
  - Python:
    - [`list.sort(*, key=None, reverse=False)`](https://docs.python.org/3/library/stdtypes.html#list.sort) stably sorts the list in place.
    - [`sorted(iterable, *, key=None, reverse=False)`](https://docs.python.org/3/library/functions.html#sorted) returns a new stably sorted list from the items in `iterable`.

### Quicksort

- Princeton
  - Text: [Sect-2.3  Quicksort](https://algs4.cs.princeton.edu/23quicksort)
  - Video: [Part-1/Week-3  Quicksort](https://www.coursera.org/learn/algorithms-part1/supplement/efbDN/lecture-slides)
- MIT
  - Text: Chap-7  Quicksort
- Libraries
  - C: [`qsort`](https://en.cppreference.com/w/c/algorithm/qsort) in `<stdlib.h>`
  - C++: [`std::sort`](https://en.cppreference.com/w/cpp/algorithm/sort) in `<algorithm>`
  - Java8: [`java.util.Arrays.sort(int[] a)`](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#sort-int:A-)

### Application: Convex Hull

- VisuAlgo
  - [Convex Hull](https://visualgo.net/en/convexhull)
- Princeton
  - Video: [Part-1/Week-2  Convex Hull](https://www.coursera.org/learn/algorithms-part1/lecture/KHJ1t/convex-hull)
- MIT
  - Text: Sect-33.3  Finding the convex hull

### Application: Collinear Points

- Princeton
  - Text: [Sect-2.5  Sorting Applications](https://algs4.cs.princeton.edu/25applications)
  - Programming Assignment: [Part-1/Week-3  Collinear Points](https://www.coursera.org/learn/algorithms-part1/programming/prXiW/collinear-points)

## Searching

### Sequential Search

### Binary Search

## Union--Find

- VisuAlgo
  - [Union--Find DS](https://visualgo.net/en/ufds)
- Princeton
  - Text: [Sect-1.5  Union--Find](https://algs4.cs.princeton.edu/15uf/)
  - Video: [Part-1/Week-1  Union--Find](https://www.coursera.org/learn/algorithms-part1/supplement/bcelg/lecture-slides)
  - Programming Assignment: [Part-1/Week-1  Percolation](https://www.coursera.org/learn/algorithms-part1/programming/Lhp5z/percolation)

## Vectors (Resizing Arrays)

- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Resizing Arrays](https://www.coursera.org/learn/algorithms-part1/lecture/WTFO7/resizing-arrays)
- MIT
  - Text: Sect-17.4  Dynamic tables
  - Video: [6.006/Lecture 9: Table Doubling, <del>Karp-Rabin</del>](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-9-table-doubling-karp-rabin)

# Lists, Stacks & Queues

## Linked Lists

- VisuAlgo
  - [Linked List](https://visualgo.net/en/list)
    - Linked List
    - Doubly Linked List
- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Stacks](https://algs4.cs.princeton.edu/13stacks/)
- MIT
  - Text: Sect-10.2  Linked lists

## Stacks

- VisuAlgo
  - [Linked List](https://visualgo.net/en/list)
    - Stack
- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Stacks](https://algs4.cs.princeton.edu/13stacks/)
- MIT
  - Text: Sect-10.1  Stacks and queues

## Queues

- VisuAlgo
  - [Linked List](https://visualgo.net/en/list)
    - Queue
    - Deque
- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Queues](https://www.coursera.org/learn/algorithms-part1/lecture/5vgrm/queues)
  - Programming Assignment: [Part-1/Week-2  Deques and Randomized Queues](https://www.coursera.org/learn/algorithms-part1/programming/zamjZ/deques-and-randomized-queues)
- MIT
  - Text: Sect-10.1  Stacks and queues

# Priority Queues

## Binary Heaps

- VisuAlgo
  - [Binary Heap](https://visualgo.net/en/heap)
- Princeton
  - Section: [Sect-2.4 Priority Queues](https://algs4.cs.princeton.edu/24pq)
  - Video: [Part-1/Week-4  Priority Queues](https://www.coursera.org/learn/algorithms-part1/supplement/eHe3d/lecture-slides)
  - Programming Assignment: [Part-1/Week-4  8-Puzzle](https://www.coursera.org/learn/algorithms-part1/programming/iqOQi/8-puzzle)
- MIT
  - Text: Chap-6  Heapsort

## Fibonacci Heaps

- MIT
  - Text: Chap-19  Fibonacci Heaps

## van Emde Boas Trees

- MIT
  - Text: Chap-20  van Emde Boas Trees

## Application: Event-Driven Simulation

- Princeton
  - Text: [Sect-6.1  Event Driven Simulation](https://algs4.cs.princeton.edu/61event)
  - Video: [Part-1/Week-4  Event-Driven Simulation](https://www.coursera.org/learn/algorithms-part1/lecture/QVhGs/event-driven-simulation-optional)
  - Code: [`CollisionSystem.java`](https://algs4.cs.princeton.edu/61event/CollisionSystem.java.html)

## Application: A* Seach for 8-Puzzle

# Hash Tables

- VisuAlgo
  - [Hash Table](https://visualgo.net/en/hashtable)
- Princeton
  - Text: [Sect-3.4  Hash Tables](https://algs4.cs.princeton.edu/34hash/)
  - Video: [Part-1/Week-6  Hash Tables](https://www.coursera.org/learn/algorithms-part1/supplement/py6zN/lecture-slides) and [Part-1/Week-6  Symbol Table Applications](https://www.coursera.org/learn/algorithms-part1/supplement/eVEjz/lecture-slides)
- MIT
  - Text: Chap-11  Hash Tables
  - Video: [6.006/Lecture 8: Hashing with Chaining](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-8-hashing-with-chaining) and [6.006/Lecture 10: Open Addressing, Cryptographic Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-10-open-addressing-cryptographic-hashing) and [6.046/Lecture 8: Randomization: Universal & Perfect Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-videos/lecture-8-randomization-universal-perfect-hashing)

According to [MIT/6.046/Lecture 8: Randomization: Universal & Perfect Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-notes/MIT6_046JS15_lec08.pdf),

> the English `hash` (1650s) means *cut into small pieces*,
> which comes from the French `hacher` which means *chop up*,
> which comes from the Old French `hache` which means *axe* (cf. English `hatchet`).

## Hash Functions

### Requirements

A good hash function should

- be *deterministic*, i.e. (after choosing the hash function,) equal keys must produce the same hash value.
- be *efficient* to compute.
- *uniformly* distribute the keys among the index range $\mathbb{Z}\cap[0, M)$.

|       Assumption       |            Given            |        Expected to be $\le1/M$        |
| :--------------------: | :-------------------------: | :-----------------------------------: |
| simple uniform hashing |    the hash function $h$    | $ \Pr_{k_1\ne k_2}\{h(k_1)=h(k_2)\} $ |
|   universal hashing    | two distinct keys $k_1,k_2$ |    $\Pr_{h\in H}\{h(k_1)=h(k_2)\}$    |

### Modular Hashing

- `int`: choose a prime `M` close to the table size and use `(key % M)` as index.
- `float`: use modular hashing on `key`'s binary representation.
- `string`: choose a small prime `R` (e.g. `31`) and do modular hashing repeatedly, i.e.
  
  ```java
  int hash = 0;
  for (int i = 0; i < key.length(); i++)
    hash = (R * hash + key.charAt(i)) % M;
  ```

### Multiplication Hashing

Suppose the `key` is a `w`-bit integer, the hash value could be obtained by the following steps:

1. Choose an integer `s` randomly from `[0, 1 << w)`.
2. Multiply the `key` by `s` to get a `2w`-bit integer, whose value is `(high << w + low)`.
3. Use the value of `low >> (w - p)`, i.e. the integer represented by the highest `p` bits of the (unsigned) `w`-bit integer `low`, as the hash value of `key`.

### Programming

If you want to make `Key` a hashable type, then do the following:

- Java: implement a method called [`hashCode()`](https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html#hashCode--), which returns a 32-bit `int`.
- Python: implement a method called  [`__hash__()`](https://docs.python.org/3/reference/datamodel.html#object.__hash__), so that the [`hash()`](https://docs.python.org/3/library/functions.html?highlight=hash%20function#hash) built-in function can be called on `Key`'s objects.
- C++: specialize the [`std::hash`](https://en.cppreference.com/w/cpp/utility/hash) template for  `Key`, so that `Key` can be used as in `std::unordered_set<Key>` or `std::unordered_map<Key, Value>`.

## Collision Resolution

### Seperate Chaining

- MIT
  - Text: Sect-11.2 Hash tables
  - Video: [6.006/Lecture 8: Hashing with Chaining](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-8-hashing-with-chaining)

- Structure: build a linked list for each array index.
- Searching: *hash to find the list* that could contain the key, then *sequentially search through that list* for the key.
- Performance: in a separate-chaining hash table with $M$ lists and $N$ keys,
  - $\Pr(\text{number of keys in a list} \approx N/M)\approx 1$, given $N/M \approx 1$.
  - $\Pr(\text{number of compares for search and insert})\propto N/M$.

### Open Addressing

- MIT
  - Text: Sect-11.4  Open addressing
  - Video:  [6.006/Lecture 10: Open Addressing, Cryptographic Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-10-open-addressing-cryptographic-hashing)

- Structure: use an array of size $M$, which is larger than $N$ --- the number of keys to be inserted.
- Searching: when there is a collision, do any of the following:
  - Linear Probing:
    - $h(k, i) = (h_1(k) + i) \bmod M$
    - i.e. check the next entry in the table (by incrementing the index) until search hit.
  - Double Hashing:
    - $h(k, i) = (h_1(k) + i \cdot h_2(k)) \bmod M$
    - The value $h_2(k)$ must be *relatively prime* to the hash-table size $M$ for the entire hash table to be searched. A convenient way to ensure this condition is
      - to let $M=2^m$ for some integer $m$, and
      - to design $h_2$ to be an odd-valued function.
- Performance: the average number of probes is
  - $\frac12\left(1 + (1 - N/M)^{-1}\right)$ for search hits.
  - $\frac12\left(1 +(1 - N/M)^{-2}\right)$ for search misses or inserts.

## Universal Hashing<a href name="uni-hash"></a>

- MIT
  - Text: Sect-11.3.3  Universal hashing
  - Video: [6.046/Lecture 8: Randomization: Universal & Perfect Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-videos/lecture-8-randomization-universal-perfect-hashing) (until 55:38)

### Universality

The collection of hash functions $H\coloneqq\{h:U\to\mathbb{Z}\cap[0,M)\}$ is said to be ***universal*** if

$$
\Pr_{h\in H}\{h(k_1)=h(k_2)\} < 1/M\qquad \forall(k_1,k_2)\in U\times (U\setminus\{k_1\})
$$

(Theorem) For $N$ arbitrary distinct keys, $M$ slots, and a random $h ∈ H$, where $H$ is a universal hashing family, there are
- $ \langle\text{#keys colliding in a slot}\rangle \le 1 + N/M $.
- $O(1+N/M)$ expected time for `Search()`, `Insert()` and `Delete()` operations.

### The Dot-product Family

Given the number of slots $M$, which is a prime, a hash function could be defined as

$$
h_{a}(k)\coloneqq(\underline{a}\cdot\underline{k})\bmod M=\left(\sum_{i=0}^{r-1}a_i k_i\right)\bmod M
$$

in which,
- $\underline{k}\coloneqq(k_0,k_1,\dots,k_{r-1})$ is the $r$-element array representation of the $k$, i.e. $k=\sum_{i=0}^{r-1} k_i M^{i}$ and
- $\underline{a}\coloneqq(a_0,a_1,\dots,a_{r-1})$ is a fixed $r$-element array, whose elements are (randomly) chosen from $\mathbb{Z}\cap[0,M)$.

### The Double-mod Family

Given the number of slots $M$, which *need not* be a prime, a hash function could be defined as

$$
h_{ab}(k)\coloneqq((ak+b)\bmod p)\bmod M
$$

in which,
- $p$ is a prime larger than $\vert U\vert$, and
- $a$ is an integer (randomly) chosen from $[1,p)$, and
- $b$ is an integer (randomly) chosen from $[0,p)$.

## Perfect Hashing

- MIT
  - Text: Sect-11.5  Perfect hashing
  - Video: [6.046/Lecture 8: Randomization: Universal & Perfect Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-videos/lecture-8-randomization-universal-perfect-hashing) (from 55:38)

### Static Dictionary

Given $N$ keys to be stored in the table, only the `Search()` operation is needed.

Requirements:

- $O(N\lg N)$ time to build with high probability.
- $O(1)$ time for `Search()` in worst case.
- $O(N)$ space in worst case.

### Two-Level Hashing

1. Hash all items with chaining using $\tilde{h}$, which is randomly chosen from a [universal hashing](#uni-hash) family.
  - If $\sum_{j=0}^{M-1}l_j^2>cN$ for a pre-chosen constant $c$, then re-choose $\tilde{h}$. 
2. For each $j \in \mathbb{Z}\cap[0,M)$, let $l_j$ be the number of items in slot $j$.
  - Pick $\tilde{\tilde{h}}_j\colon U \to \mathbb{Z}\cap[0,M_j)$ from a [universal hashing](#uni-hash) family for $M_j \in[l_j^2, O(l_j^2)]$.
  - Replace the chain in slot $j$ with a hash table using $\tilde{\tilde{h}}_j$.
  - If $\tilde{\tilde{h}}_j(k_r)=\tilde{\tilde{h}}_j(k_s)$ for any $r\ne s$, then repick $\tilde{\tilde{h}}_j$ and rehash those $l_j$ items.

# Trees

- MIT
  - Text: Sect-B.5  Trees

## Binary Search Trees

- VisuAlgo
  - [Binary Search Tree](https://visualgo.net/en/bst)
- Princeton
  - Text: [Sect-3.1  Symbol Tables](https://algs4.cs.princeton.edu/31elementary) and [Sect-3.2  Binary Search Trees](https://algs4.cs.princeton.edu/32bst)
  - Video: [Part-1/Week-4  Elementary Symbol Tables](https://www.coursera.org/learn/algorithms-part1/supplement/2kwpU/lecture-slides)
- MIT
  - Text: Chap-12  Binary Search Trees

## Balanced Search Trees

- Princeton
  - Text: [Sect-3.3 Balanced Search Trees](https://algs4.cs.princeton.edu/33balanced) and [Sect-6.2 B-trees](https://algs4.cs.princeton.edu/62btree)
  - Video: [Part-1/Week-5  Balanced Search Trees](https://www.coursera.org/learn/algorithms-part1/supplement/zQXMd/lecture-slides)

### AVL Trees

### Red-Black Trees

- MIT
  - Text: Chap-13  Red-Black Trees

### B-trees

- MIT
  - Text: Chap-18  B-Trees

## Geometric Search Trees

### Kd-Trees

- Princeton
  - Video: [Part-1/Week-5  Geometric Applications of BSTs](https://www.coursera.org/learn/algorithms-part1/supplement/yelcJ/lecture-slides)
  - Programming Assignment: [Part-1/Week-5  Kd-Trees](https://www.coursera.org/learn/algorithms-part1/programming/wuF0a/kd-trees)

### Range Trees

# Graphs

- MIT
  - Text: Sect-B.4  Graphs

## Graph Structures

- VisuAlgo: [Graph Structures](https://visualgo.net/en/graphds) and [Graph Traversal](https://visualgo.net/en/dfsbfs)
- Princeton
  - Text: [Sect-4.1  Undirected Graphs](https://algs4.cs.princeton.edu/41graph) and [Sect-4.2  Directed Graphs](https://algs4.cs.princeton.edu/42digraph)
  - Video: [Part-2/Week-1  Undirected Graphs](https://www.coursera.org/learn/algorithms-part2/supplement/NlsQF/lecture-slides) and [Part-2/Week-1  Directed Graphs](https://www.coursera.org/learn/algorithms-part2/supplement/qRjk3/lecture-slides)
  - Programming Assignment: [WordNet](https://www.coursera.org/learn/algorithms-part2/programming/BCNsp/wordnet)

### Breadth-First Search

### Depth-First Search

### Topological Sort

- Libraries
  - Python3: [`graphlib.TopologicalSorter`](https://docs.python.org/3/library/graphlib.html#graphlib.TopologicalSorter) provides functionality to topologically sort a graph of hashable nodes.

## Minimum Spanning Trees

- VisuAlgo
  - [Min Spanning Tree](https://visualgo.net/en/mst)
- Princeton
  - Text: [Sect-4.3  Minimum Spanning Trees](https://algs4.cs.princeton.edu/43mst)
  - Video: [Part-2/Week-2  Minimun Spanning Trees](https://www.coursera.org/learn/algorithms-part2/supplement/tda2O/lecture-slides)

### Kruskal's Algorithm

### Prim's Algorithm

## Shortest Paths

- VisuAlgo
  - [Single-Source Shortest Paths](https://visualgo.net/en/sssp)
- Princeton
  - Text: [Sect-4.4  Shortest Paths](https://algs4.cs.princeton.edu/44sp)
  - Video: [Part-2/Week-2  Shortest Paths](https://www.coursera.org/learn/algorithms-part2/supplement/BZTAt/lecture-slides)
  - Programming Assignment: [Seam Carving](https://www.coursera.org/learn/algorithms-part2/programming/cOdkz/seam-carving)

### Dijkstra's Algorithm

## Maximum Flow and Minimum Cut

- VisuAlgo
  - [Network Flow](https://visualgo.net/en/maxflow)
- Princeton
  - Text: [Sect-6.4  Maxflow](https://algs4.cs.princeton.edu/64maxflow)
  - Video: [Part-2/Week-3  Maximum Flow and Minimum Cut](https://www.coursera.org/learn/algorithms-part2/supplement/qKIDx/lecture-slides)
  - Programming Assignment: [Baseball Elimination](https://www.coursera.org/learn/algorithms-part2/programming/hmYRI/baseball-elimination)

### Ford--Fulkerson Algorithm

### Maxflow--Mincut Theorem

# Strings

## String Implementations

- Princeton
  - Text: [Sect-5.0  Overview](https://algs4.cs.princeton.edu/50strings)
  - Video: [Part-2/Week-3  Strings in Java](https://www.coursera.org/learn/algorithms-part2/lecture/vGHvb/strings-in-java)
- Libraries
  - Java8: [`java.lang.String`](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html) and [`java.lang.StringBuilder`](https://docs.oracle.com/javase/8/docs/api/java/lang/StringBuilder.html)
  - Python3: [`str`](https://docs.python.org/3/library/stdtypes.html#str) (Python class, in Built-in Types) and [`string`](https://docs.python.org/3/library/string.html#module-string) (Python module, in `string` — Common string operations)
  - C++: [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string) in [`<string>`](https://en.cppreference.com/w/cpp/header/string)
  - C: [Null-terminated byte strings](https://en.cppreference.com/w/c/string/byte) in `<string.h>`

## Radix Sorts

- Princeton
  - Text: [Sect-5.1  String Sorts](https://algs4.cs.princeton.edu/51radix)
  - Video: [Part-2/Week-3  Radix Sorts](https://www.coursera.org/learn/algorithms-part2/supplement/v5gBy/lecture-slides)
- MIT
  - Text: Sect-8.3  Radix sort

### LSD Radix Sort

### MSD Radix Sort

### 3-way Radix Quicksort

## Suffix Arrays

- VisuAlgo
  - [Suffix Array](https://visualgo.net/en/suffixarray)
- Princeton
  - Text: [Sect-6.3  Suffix Arrays](https://algs4.cs.princeton.edu/63suffix)
  - Video: [Part-2/Week-3  Suffix Arrays](https://www.coursera.org/learn/algorithms-part2/lecture/TH18W/suffix-arrays)

## Tries

- Princeton
  - Text: [Sect-5.2  Tries](https://algs4.cs.princeton.edu/52trie)
  - Video: [Part-2/Week-4  Tries](https://www.coursera.org/learn/algorithms-part2/supplement/LW7HJ/lecture-slides)

### R-way Tries

### Ternary Search Tries

## Substring Search

- Princeton
  - Text: [Sect-5.3  Substring Search](https://algs4.cs.princeton.edu/53substring)
  - Video: [Part-2/Week-4  Substring Search](https://www.coursera.org/learn/algorithms-part2/supplement/CrTCF/lecture-slides)
  - Programming Assignment: [Boggle](https://www.coursera.org/learn/algorithms-part2/programming/9GqJs/boggle)
- MIT
  - Text: Chap-32  String Matching

### Knuth--Morris--Pratt (KMP)

#### Deterministic Finite-state Automaton (DFA)

### Boyer--More

### Rabin--Karp

- MIT
  - Video: [6.006/Lecture 9: <del>Table Doubling</del>, Karp-Rabin](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-9-table-doubling-karp-rabin)

## Regular Expressions

- Princeton
  - Text: [Sect-5.4  Regular Expressions](https://algs4.cs.princeton.edu/54regexp)
  - Video: [Part-2/Week-5  Regular Expressions](https://www.coursera.org/learn/algorithms-part2/supplement/dBpZD/lecture-slides)
- Libraries
  - Java8: [`java.util.regex`](https://docs.oracle.com/javase/8/docs/api/java/util/regex/package-summary.html)
  - Python3: [`re`](https://docs.python.org/3/library/re.html) (Regular expression operations) and [Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html)
  - C++: [`std::regex`](https://en.cppreference.com/w/cpp/regex/basic_regex) in [`<regex>`](https://en.cppreference.com/w/cpp/header/regex)
  - Visual Studio: [Use regular expressions in Visual Studio](https://docs.microsoft.com/en-us/visualstudio/ide/using-regular-expressions-in-visual-studio)
  - Shell: [正規表示法與文件格式化處理](http://linux.vbird.org/linux_basic/0330regularex.php) and [Regular Expressions](https://www.gnu.org/software/grep/manual/html_node/Regular-Expressions.html)

### Nondeterministic Finite-state Automaton (NFA)

## Data Compression

- Princeton
  - Text: [Sect-5.5  Data Compression](https://algs4.cs.princeton.edu/55compression)
  - Video: [Part-2/Week-5  Data Compression](https://www.coursera.org/learn/algorithms-part2/supplement/gRhgE/lecture-slides)
  - Programming Assignment: [Burrows Wheeler](https://www.coursera.org/learn/algorithms-part2/programming/3nmSB/burrows-wheeler)
