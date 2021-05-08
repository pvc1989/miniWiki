---
title: 虚拟内存
---

# 1. 物理与虚拟地址

# 2. 地址空间

# 3. 虚拟内存的缓存功能

# 4. 虚拟内存的管理功能

# 5. 虚拟内存的保护功能

# 6. 地址转换

# 7. 实例：Intel Core i7 / Linux 内存系统

# 8. 内存映射<a href id="memory-map"></a>

# 9. 动态内存分配

## 9.1. `malloc()` 与 `free()`

## 9.2. 为何需要动态内存？

## 9.3. 分配器的需求与目标

## 9.4. 碎片化

## 9.5. 实现难点

## 9.6. 隐式的自由块链表

## 9.7. 放置被分配块

## 9.8. 分裂过大的自由块

## 9.9. 获取更多的堆内存

## 9.10. 合并相邻的自由块

## 9.11. 利用边界标签合并

## 9.12. 简易分配器的实现

### 设计概要

`memlib.c` 实现了一个简易的虚拟内存系统：

```c
#define MAX_HEAP (20*(1<<20))  /* 20 MB */

static char *mem_head;     /* 堆首字节的地址 */
static char *mem_tail;      /* 堆尾字节的地址 + 1 */
static char *mem_max_addr; /* 最大合法堆地址 + 1 */

void mem_init(void) {
    mem_head = (char *)Malloc(MAX_HEAP);
    mem_tail = (char *)mem_head;
    mem_max_addr = (char *)(mem_head + MAX_HEAP);
}

void *mem_sbrk(int incr) {
    char *old_brk = mem_tail;
    if ((incr < 0) || ((mem_tail + incr) > mem_max_addr)) {
        errno = ENOMEM;
        fprintf(stderr, "ERROR: mem_sbrk failed. Ran out of memory...\n");
        return (void *)-1;
    }
    mem_tail += incr;
    return (void *)old_brk;
}
```

`mm.c` 实现了一个基于“隐式链表”的分配器：

```c
extern int mm_init(void);
extern void *mm_malloc(size_t size);
extern void mm_free(void *ptr);
```

为简化“合并相邻的自由块”，在链表两端引入一对辅助块：
- 【起始块 (prologue block)】长度为 8 B，只含“头标”及“脚标”。
- 【结尾块 (epilogue block)】长度为 4 B，只含“头标”，并标记为“已分配的”。

### 尺寸常量、链表操作宏

```c
#define WSIZE       4   /* word size (bytes) */
#define DSIZE       8   /* double word size (bytes) */
#define CHUNK  (1<<12)  /* extend heap by this amount (bytes) */

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* PACK a size and allocated bit into a word */
#define PACK(size, alloc)  ((size) | (alloc))

/* GET and PUT a word at address p */
#define GET(p)       (*(unsigned int *)(p))
#define PUT(p, val)  (*(unsigned int *)(p) = (val))

/* GET the SIZE and ALLOCated fields from address p */
#define GET_SIZE(p)   (GET(p) & ~0x7)
#define GET_ALLOC(p)  (GET(p) &  0x1)

/* Given block ptr block, compute address of its HEADER and FOOTER */
#define HEADER(block)  ((char *)(block) - WSIZE)
#define FOOTER(block)  ((char *)(block) + GET_SIZE(HEADER(block)) - DSIZE)

/* Given block ptr block, compute address of NEXT and PREVious blocks */
#define NEXT(block)  ((char *)(block) + GET_SIZE(((char *)(block) - WSIZE)))
#define PREV(block)  ((char *)(block) - GET_SIZE(((char *)(block) - DSIZE)))

/* e.g. get the size of the next block */
size_t size = GET_SIZE(HEADER(NEXT(block)));
```

### 链表初始化

```c
/* Global variables */
static char *first_block = NULL; /* Pointer to first block */

int mm_init(void) {
    /* Create the initial empty heap */
    if ((first_block = mem_sbrk(4*WSIZE)) == (void *)-1)
        return -1;
    PUT(first_block, 0);                          /* Padding on head */
    PUT(first_block + (1*WSIZE), PACK(DSIZE, 1)); /* Prologue header */ 
    PUT(first_block + (2*WSIZE), PACK(DSIZE, 1)); /* Prologue footer */ 
    PUT(first_block + (3*WSIZE), PACK(0, 1));     /* Epilogue header */
    first_block += (2*WSIZE);

    /* Extend the empty heap with a free block of CHUNK bytes */
    if (extend_heap(CHUNK / WSIZE) == NULL) 
        return -1;
    return 0;
}

static void *extend_heap(size_t words) {
    char *block;
    size_t size;

    /* Allocate an even number of words to maintain alignment */
    size = ((words % 2) + words) * WSIZE;
    if ((block = mem_sbrk(size)) == (void *)-1)  
        return NULL;

    /* Initialize free block header/footer and the epilogue header */
    PUT(HEADER(block), PACK(size, 0));         /* Free block header */
    PUT(FOOTER(block), PACK(size, 0));         /* Free block footer */
    PUT(HEADER(NEXT(block)), PACK(0, 1));    /* New epilogue header */

    /* Coalesce if the previous block was free */
    return coalesce(block);
}
```

### 块的释放与合并

```c
void mm_free(void *block) {
    if (block == NULL) 
        return;

    size_t size = GET_SIZE(HEADER(block));
    if (first_block == NULL)
        mm_init();

    PUT(HEADER(block), PACK(size, 0));
    PUT(FOOTER(block), PACK(size, 0));
    coalesce(block);
}

static void *coalesce(void *block) {
    size_t prev_alloc = GET_ALLOC(FOOTER(PREV(block)));
    size_t next_alloc = GET_ALLOC(HEADER(NEXT(block)));
    size_t size = GET_SIZE(HEADER(block));

    if (prev_alloc && next_alloc) {            /* Case 1 */
    }
    else if (prev_alloc && !next_alloc) {      /* Case 2 */
        size += GET_SIZE(HEADER(NEXT(block)));
        PUT(HEADER(block), PACK(size, 0));
        PUT(FOOTER(block), PACK(size,0));
    }
    else if (!prev_alloc && next_alloc) {      /* Case 3 */
        size += GET_SIZE(HEADER(PREV(block)));
        PUT(FOOTER(block), PACK(size, 0));
        PUT(HEADER(PREV(block)), PACK(size, 0));
        block = PREV(block);
    }
    else {                                     /* Case 4 */
        size += GET_SIZE(HEADER(PREV(block)))
              + GET_SIZE(FOOTER(NEXT(block)));
        PUT(HEADER(PREV(block)), PACK(size, 0));
        PUT(FOOTER(NEXT(block)), PACK(size, 0));
        block = PREV(block);
    }

    return block;
}
```

### 块的分配

```c
void *mm_malloc(size_t size) {
    size_t alloc_size; /* Adjusted block size */
    size_t chunk_size; /* Amount to extend heap if no fit */
    char *block;

    if (first_block == NULL)
        mm_init();

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    if (size <= DSIZE)
        alloc_size = DSIZE + DSIZE; /* payload + overhead */
    else
        alloc_size = DSIZE * (((size + DSIZE) + (DSIZE-1)) / DSIZE);

    /* Search the free list for a fit */
    if ((block = find_fit(alloc_size)) == NULL) {
        /* No fit found. Get more memory. */
        chunk_size = MAX(alloc_size, CHUNK);
        if ((block = extend_heap(chunk_size/WSIZE)) == NULL)  
            return NULL;
    }
    place(block, alloc_size);
    return block;
}

static void *find_fit(size_t alloc_size) {
    char *block = first_block;
    size_t block_size; /* Size of current block */

    do { /* First-hit */
        block = NEXT(block);
        block_size = GET_SIZE(HEADER(block));
        if (block_size == 0) {
            block = NULL;
            break;
        }
    } while (block_size < alloc_size || GET_ALLOC(HEADER(block)));

    return block;
}

static void place(void *block, size_t alloc_size) {
    size_t block_size = GET_SIZE(HEADER(block));
    int split = (block_size > alloc_size + QSIZE);

    if (split) {
        /* The remaining of current block can hold a min-sized block. */
        PUT(HEADER(block), PACK(alloc_size, 1));
        PUT(FOOTER(block), PACK(alloc_size, 1));
        block = NEXT(block);
        block_size -= alloc_size;
    }
    PUT(HEADER(block), PACK(block_size, !split));
    PUT(FOOTER(block), PACK(block_size, !split));
}
```

## 9.13. 显式的自由块链表

自由块的“有效载荷”部分可用于存储指针，以便构造出“链表”等数据结构。

释放策略：

- 【后进先出 (LIFO)】刚被释放的块总是被置于链表头部。
- 【地址顺序】链表中的（自由）块总是按地址升序排列。

## 9.14. 分离的自由块链表

【分离链表 (segregated lists)】将自由块组织为多条链表，每条链表中的自由块具有大致相同的尺寸。

- $(0,1],(1,2],(2,4],\dots,(512,1024],(1024,2048],(2048,4096],(4096,\infty)$
- $(0,1],\dots,(1022,1023],(1023,1024],(1024,2048],(2048,4096],(4096,\infty)$

### 简易分离存储

【简易分离存储 (simple segregated storage)】每条链表中的自由块具有相同的尺寸；不分裂或合并自由块。

- 【优点】分配、释放均可在常数时间内完成（调用 `sbrk()` 除外）；无需头标、脚标；用单向链表即可实现。
- 【缺点】空间利用率低。

### 分离匹配

【分离匹配 (segregated fits)】GCC 的实现方案。

- 【分配】
  - 根据所需块的尺寸，确定最匹配的链表，在其中查找首个可用的自由块；若找不到，则进入自由块尺寸更大的链表查找。
  - 若所有链表均无可用自由块，则向操作系统请求更多堆内存。
  - 分割找到的自由块，将剩余部分插入合适的链表。
- 【释放】
  - 若可能，合并相邻自由块（四种情形）。
  - 将所得（可能更大的）自由块插入合适的链表。

# 10. 垃圾回收

# 11. 与内存相关的代码缺陷