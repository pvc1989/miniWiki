---
title: 散列 (Hashing)
---

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

# 散列函数

## 必要条件

A good hash function should

- be *deterministic*, i.e. (after choosing the hash function,) equal keys must produce the same hash value.
- be *efficient* to compute.
- *uniformly* distribute the keys among the index range $\mathbb{Z}\cap[0, M)$.

|       Assumption       |            Given            |        Expected to be $\le1/M$        |
| :--------------------: | :-------------------------: | :-----------------------------------: |
| simple uniform hashing |    the hash function $h$    | $ \Pr_{k_1\ne k_2}\{h(k_1)=h(k_2)\} $ |
|   universal hashing    | two distinct keys $k_1,k_2$ |    $\Pr_{h\in H}\{h(k_1)=h(k_2)\}$    |

## 基于余数的散列函数

- `int`: choose a prime `M` close to the table size and use `(key % M)` as index.
- `float`: use modular hashing on `key`'s binary representation.
- `string`: choose a small prime `R` (e.g. `31`) and do modular hashing repeatedly, i.e.
  
  ```java
  int hash = 0;
  for (int i = 0; i < key.length(); i++)
    hash = (R * hash + key.charAt(i)) % M;
  ```

## 基于乘法的散列函数

Suppose the `key` is a `w`-bit integer, the hash value could be obtained by the following steps:

1. Choose an integer `s` randomly from `[0, 1 << w)`.
2. Multiply the `key` by `s` to get a `2w`-bit integer, whose value is `(high << w + low)`.
3. Use the value of `low >> (w - p)`, i.e. the integer represented by the highest `p` bits of the (unsigned) `w`-bit integer `low`, as the hash value of `key`.

## 程序实现

If you want to make `Key` a hashable type, then do the following:

- Java: implement a method called [`hashCode()`](https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html#hashCode--), which returns a 32-bit `int`.
- Python: implement a method called  [`__hash__()`](https://docs.python.org/3/reference/datamodel.html#object.__hash__), so that the [`hash()`](https://docs.python.org/3/library/functions.html?highlight=hash%20function#hash) built-in function can be called on `Key`'s objects.
- C++: specialize the [`std::hash`](https://en.cppreference.com/w/cpp/utility/hash) template for  `Key`, so that `Key` can be used as in `std::unordered_set<Key>` or `std::unordered_map<Key, Value>`.

# 冲突化解

## 分离链法

- MIT
  - Text: Sect-11.2 Hash tables
  - Video: [6.006/Lecture 8: Hashing with Chaining](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-8-hashing-with-chaining)

- Structure: build a linked list for each array index.
- Searching: *hash to find the list* that could contain the key, then *sequentially search through that list* for the key.
- Performance: in a separate-chaining hash table with $M$ lists and $N$ keys,
  - $\Pr(\text{number of keys in a list} \approx N/M)\approx 1$, given $N/M \approx 1$.
  - $\Pr(\text{number of compares for search and insert})\propto N/M$.

## 开地址法

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

# 全域散列

- MIT
  - Text: Sect-11.3.3  Universal hashing
  - Video: [6.046/Lecture 8: Randomization: Universal & Perfect Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-videos/lecture-8-randomization-universal-perfect-hashing) (until 55:38)

## 全域散列函数族

The collection of hash functions $H\coloneqq\{h:U\to\mathbb{Z}\cap[0,M)\}$ is said to be ***universal*** if

$$
\Pr_{h\in H}\{h(k_1)=h(k_2)\} < 1/M\qquad \forall(k_1,k_2)\in U\times (U\setminus\{k_1\})
$$

(Theorem) For $N$ arbitrary distinct keys, $M$ slots, and a random $h ∈ H$, where $H$ is a universal hashing family, there are
- $ \langle\text{#keys colliding in a slot}\rangle \le 1 + N/M $.
- $O(1+N/M)$ expected time for `Search()`, `Insert()` and `Delete()` operations.

## 基于点乘的全域散列

Given the number of slots $M$, which is a prime, a hash function could be defined as

$$
h_{a}(k)\coloneqq(\underline{a}\cdot\underline{k})\bmod M=\left(\sum_{i=0}^{r-1}a_i k_i\right)\bmod M
$$

in which,
- $\underline{k}\coloneqq(k_0,k_1,\dots,k_{r-1})$ is the $r$-element array representation of the $k$, i.e. $k=\sum_{i=0}^{r-1} k_i M^{i}$ and
- $\underline{a}\coloneqq(a_0,a_1,\dots,a_{r-1})$ is a fixed $r$-element array, whose elements are (randomly) chosen from $\mathbb{Z}\cap[0,M)$.

## 基于余数的全域散列

Given the number of slots $M$, which *need not* be a prime, a hash function could be defined as

$$
h_{ab}(k)\coloneqq((ak+b)\bmod p)\bmod M
$$

in which,
- $p$ is a prime larger than $\vert U\vert$, and
- $a$ is an integer (randomly) chosen from $[1,p)$, and
- $b$ is an integer (randomly) chosen from $[0,p)$.

# 完美散列

- MIT
  - Text: Sect-11.5  Perfect hashing
  - Video: [6.046/Lecture 8: Randomization: Universal & Perfect Hashing](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-046j-design-and-analysis-of-algorithms-spring-2015/lecture-videos/lecture-8-randomization-universal-perfect-hashing) (from 55:38)

## 静态字典

Given $N$ keys to be stored in the table, only the `Search()` operation is needed.

Requirements:

- $O(N\lg N)$ time to build with high probability.
- $O(1)$ time for `Search()` in worst case.
- $O(N)$ space in worst case.

## 二重散列

1. Hash all items with chaining using $\tilde{h}$, which is randomly chosen from a [universal hashing](#全域散列) family.
  - If $\sum_{j=0}^{M-1}l_j^2>cN$ for a pre-chosen constant $c$, then re-choose $\tilde{h}$. 
2. For each $j \in \mathbb{Z}\cap[0,M)$, let $l_j$ be the number of items in slot $j$.
  - Pick $\tilde{\tilde{h}}_j\colon U \to \mathbb{Z}\cap[0,M_j)$ from a [universal hashing](#全域散列) family for $M_j \in[l_j^2, O(l_j^2)]$.
  - Replace the chain in slot $j$ with a hash table using $\tilde{\tilde{h}}_j$.
  - If $\tilde{\tilde{h}}_j(k_r)=\tilde{\tilde{h}}_j(k_s)$ for any $r\ne s$, then repick $\tilde{\tilde{h}}_j$ and rehash those $l_j$ items.

