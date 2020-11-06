---
title: 数据结构与算法
---

# 学习资源

## 教程

### Princeton

- Textbooks:
  - [Algorithms, 4th Edition](https://algs4.cs.princeton.edu/home/)
  - [An Introduction to the Analysis of Algorithms, 2nd Edition](https://aofa.cs.princeton.edu/home/)
- Courses:
  - [Algorithms, Part 1](https://www.coursera.org/learn/algorithms-part1)
  - [Algorithms, Part 2](https://www.coursera.org/learn/algorithms-part2)
  - [Analysis of Algorithms](https://www.coursera.org/learn/analysis-of-algorithms)

### MIT

- Textbook:
  - [Introduction to Algorithms, Third Edition](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)
- Courses:
  - [Introduction to Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/) taught by *Prof. Erik Demaine* and *Prof. Srini Devadas* in Fall 2011.
  - [Design and Analysis of Algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/) taught by *Prof. Erik Demaine* and *Prof. Srini Devadas* and *Prof. Nancy Lynch* in Spring 2015.

## 网站

### 学习

- [VisuAlgo](https://visualgo.net/en): visualising data structures and algorithms through animation.

### 刷题

- [LeetCode](https://leetcode.com/): the world's leading online programming learning platform.
- [PAT](https://www.patest.cn/): Programming Ability Test.

# Computational Complexity

## Mathematical (Theoretical) Method

- MIT
  - Text:
    - Chap-3  Growth of Functions
    - Chap-5  Probabilistic Analysis and Randomized Algorithms

## Scientific (Experimental) Method

- Princeton
  - Text: [Sect-1.4  Analysis of Algorithms](https://algs4.cs.princeton.edu/14analysis/)
  - Video: [Part-1/Week-1  Analysis of Algorithms](https://www.coursera.org/learn/algorithms-part1/supplement/mpK20/lecture-slides)

> 1. *Observe* some feature of the natural world, generally with precise measurements.
> 2. *Hypothesize* a model that is consistent with the observations.
> 3. *Predict* events using the hypothesis.
> 4. *Verify* the predictions by making further observations.
> 5. *Validate* by repeating until the hypothesis and observations agree.

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

### Shellsort

- Princeton
  - Text: *Shellsort* in [Sect-2.1  Elementary Sorts](https://algs4.cs.princeton.edu/21elementary)
  - Video: [Part-1/Week-2  Shellsort](https://www.coursera.org/learn/algorithms-part1/lecture/zPYhF/shellsort)

### Mergesort

- Princeton
  - Text: [Sect-2.2  Mergesort](https://algs4.cs.princeton.edu/22mergesort)
  - Video: [Part-1/Week-3  Mergesort](https://www.coursera.org/learn/algorithms-part1/supplement/4E9fa/lecture-slides)

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
- Libraries
  - C: [`qsort`](https://en.cppreference.com/w/c/algorithm/qsort) in `<stdlib.h>`
  - C++: [`std::sort`](https://en.cppreference.com/w/cpp/algorithm/sort) in `<algorithm>`
  - Java8: [`java.util.Arrays.sort(int[] a)`](https://docs.oracle.com/javase/8/docs/api/java/util/Arrays.html#sort-int:A-)

### Application: Convex Hull

- VisuAlgo
  - [Convex Hull](https://visualgo.net/en/convexhull)
- Princeton
  - Video: [Part-1/Week-2  Convex Hull](https://www.coursera.org/learn/algorithms-part1/lecture/KHJ1t/convex-hull)

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

# Lists, Stacks & Queues

## Lists

- VisuAlgo
  - [Linked List](https://visualgo.net/en/list)
    - Linked List
    - Doubly Linked List
- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Stacks](https://algs4.cs.princeton.edu/13stacks/)

## Stacks

- VisuAlgo
  - [Linked List](https://visualgo.net/en/list)
    - Stack
- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Stacks](https://algs4.cs.princeton.edu/13stacks/)

## Queues

- VisuAlgo
  - [Linked List](https://visualgo.net/en/list)
    - Queue
    - Deque
- Princeton
  - Text: [Sect-1.3  Bags, Queues, and Stacks](https://algs4.cs.princeton.edu/13stacks/)
  - Video: [Part-1/Week-2  Queues](https://www.coursera.org/learn/algorithms-part1/lecture/5vgrm/queues)
  - Programming Assignment: [Part-1/Week-2  Deques and Randomized Queues](https://www.coursera.org/learn/algorithms-part1/programming/zamjZ/deques-and-randomized-queues)

# Priority Queues

## Binary Heap

- Princeton
  - Section: [Sect-2.4 Priority Queues](https://algs4.cs.princeton.edu/24pq)
  - Video: [Part-1/Week-4  Priority Queues](https://www.coursera.org/learn/algorithms-part1/supplement/eHe3d/lecture-slides)
  - Programming Assignment: [Part-1/Week-4  8-Puzzle](https://www.coursera.org/learn/algorithms-part1/programming/iqOQi/8-puzzle)

### Event-Driven Simulation

- Princeton
  - Text: [Sect-6.1  Event Driven Simulation](https://algs4.cs.princeton.edu/61event)
  - Video: [Part-1/Week-4  Event-Driven Simulation](https://www.coursera.org/learn/algorithms-part1/lecture/QVhGs/event-driven-simulation-optional)
  - Code: [`CollisionSystem.java`](https://algs4.cs.princeton.edu/61event/CollisionSystem.java.html)

## Fibonacci Heap

## Van Emde Boas Tree

# Hash Tables

- VisuAlgo
  - [Hash Table](https://visualgo.net/en/hashtable)
- Princeton
  - Text: [Sect-3.4  Hash Tables](https://algs4.cs.princeton.edu/34hash/)
  - Video: [Part-1/Week-6  Hash Tables](https://www.coursera.org/learn/algorithms-part1/supplement/py6zN/lecture-slides) and [Part-1/Week-6  Symbol Table Applications](https://www.coursera.org/learn/algorithms-part1/supplement/eVEjz/lecture-slides)

## Hash Functions

### Requirements

A good hash function should

- be *deterministic*, i.e. equal keys must produce the same hash value.
- be *efficient* to compute.
- *uniformly* distribute the keys among the index range `[0, M)`.

### Modular Hashing

- `int`s: choose a prime `M` close to the table size and use `(key % M)` as index.
- `float`s: use modular hashing on `key`'s binary representation.
- `string`s: choose a small prime `R` (e.g. `31`) and do modular hashing repeatedly, i.e.
  
  ```java
  int hash = 0;
  for (int i = 0; i < key.length(); i++)
    hash = (R * hash + key.charAt(i)) % M;
  ```

### Programming

If you want to make `K` a hashable type, then do the following:

- Java: implement a method called [`hashCode()`](https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html#hashCode--), which returns a 32-bit `int`.
- Python: implement a method called  [`__hash__()`](https://docs.python.org/3/reference/datamodel.html#object.__hash__), so that the [`hash()`](https://docs.python.org/3/library/functions.html?highlight=hash%20function#hash) built-in function can be called on `K`'s objects.
- C++: specialize the [`std::hash`](https://en.cppreference.com/w/cpp/utility/hash) template for  `K`, so that `K` can be used as in `std::unordered_set<K>` or `std::unordered_map<K, V>`.

## Collision Resolution

### Seperate Chaining

- Structure: build a linked list for each array index.
- Searching: *hash to find the list* that could contain the key, then *sequentially search through that list* for the key.
- Performance: in a separate-chaining hash table with $M$ lists and $N$ keys,
  - $\Pr(\text{number ot keys in a list} \approx N/M)\approx 1$, given $N/M \approx 1$.
  - $\Pr(\text{number of compares for search and insert})\propto N/M$.

### Open Addressing

- Structure: use an array of size $M$, which is larger than $N$ --- the number of keys to be inserted.
- Searching: when there is a collision, do any of the following:
  - Linear Probing: check the next entry in the table (by incrementing the index) until search hit.
  - Double Probing:
- Performance: the average number of probes is
  - $\frac12\left(1 + (1 - N/M)^{-1}\right)$ for search hits.
  - $\frac12\left(1 +(1 - N/M)^{-2}\right)$ for search misses or inserts.

## Universal Hashing

# Trees

## Binary Search Trees

- VisuAlgo
  - [Binary Search Tree](https://visualgo.net/en/bst)
- Princeton
  - Text: [Sect-3.1  Symbol Tables](https://algs4.cs.princeton.edu/31elementary) and [Sect-3.2  Binary Search Trees](https://algs4.cs.princeton.edu/32bst)
  - Video: [Part-1/Week-4  Elementary Symbol Tables](https://www.coursera.org/learn/algorithms-part1/supplement/2kwpU/lecture-slides)

## Balanced Search Trees

- Princeton
  - Text: [Sect-3.3 Balanced Search Trees](https://algs4.cs.princeton.edu/33balanced) and [Sect-6.2 B-trees](https://algs4.cs.princeton.edu/62btree)
  - Video: [Part-1/Week-5  Balanced Search Trees](https://www.coursera.org/learn/algorithms-part1/supplement/zQXMd/lecture-slides)

### AVL Trees

### Red-Black Trees

### B-trees

## Geometric Search Trees

### Kd-Trees

- Princeton
  - Video: [Part-1/Week-5  Geometric Applications of BSTs](https://www.coursera.org/learn/algorithms-part1/supplement/yelcJ/lecture-slides)
  - Programming Assignment: [Part-1/Week-5  Kd-Trees](https://www.coursera.org/learn/algorithms-part1/programming/wuF0a/kd-trees)

### Range Trees

# Graphs

## Undirected Graphs

## Directed Graphs

### Application: [WordNet](https://www.coursera.org/learn/algorithms-part2/programming/BCNsp/wordnet)

## Minimum Spanning Trees

## Shortest Paths

### Application: [Seam Carving](https://www.coursera.org/learn/algorithms-part2/programming/cOdkz/seam-carving)

## Maximum Flow and Minimum Cut

### Application: [Baseball Elimination](https://www.coursera.org/learn/algorithms-part2/programming/hmYRI/baseball-elimination)

# Strings

## String Implementations

## Radix Sorts

### LSD Radix Sort

### MSD Radix Sort

### 3-way Radix Quicksort

## Suffix Arrays

## Tries

## Substring Search

### Brute-Force

### Knuth--Morris--Pratt

### Boyer--More

### Rabin--Karp

### Application: [Boggle](https://www.coursera.org/learn/algorithms-part2/programming/9GqJs/boggle)

## Regular Expressions

## Data Compression

### Application: [Burrows Wheeler](https://www.coursera.org/learn/algorithms-part2/programming/3nmSB/burrows-wheeler)

# Computation Theory

## Reductions

## Linear Programming

## Intractability
