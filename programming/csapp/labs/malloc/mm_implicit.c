/*
 * mm_implicit.c - The CS:APP3e malloc package.
 *
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  Blocks are never coalesced or reused.  The size of
 * a block is found at the first aligned word before the block (we need
 * it for realloc).
 *
 * This code is correct and blazingly fast, but very bad usage-wise since
 * it never frees anything.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mm.h"
#include "memlib.h"

/* If you want debugging output, use the following macro.  When you hand
 * in, remove the #define DEBUG line. */
#ifdef DEBUG
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_checkheap() mm_checkheap(1)
#else
#define dbg_printf(...)
#define dbg_checkheap()
#endif

/* do not change the following! */
#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#endif /* def DRIVER */

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define WSIZE       4   /* single word size (bytes) */
#define DSIZE       8   /* double word size (bytes) */
#define QSIZE      16   /*  quad  word size (bytes) */
#define CHUNK  (1<<12)  /* extend heap by this amount (bytes) */

/* single word (4) or double word (8) alignment */
#define ALIGNMENT DSIZE

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT-1)) & ~0x7)

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

/* Global variables */
static char *prolog_block = NULL; /* Pointer to the prologue block */
#ifdef NEXT_FIT
static char *latest_block = NULL; /* Pointer to the latest touched block */
#endif

/* Private methods */

static void *coalesce(void *block)
{
    dbg_printf("coalesce(%p) is ready to start.\n", block);

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

    dbg_checkheap();
    dbg_printf("%p = coalesce() is ready to exit.\n", block);
    return block;
}

static void *extend_heap(size_t words)
{
    dbg_printf("extend_heap(0x%lx) is ready to start.\n", words * WSIZE);

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
    block = coalesce(block);
    dbg_printf("%p = extend_heap() is ready to exit.\n", block);
    return block;
}

static void *find_fit(size_t alloc_size)
{
    dbg_printf("find_fit(0x%lx) is ready to start.\n", alloc_size);

#ifdef NEXT_FIT
    char *block = latest_block;
#else
    char *block = prolog_block;
#endif
    size_t block_size; /* Size of current block */

    do { /* First-hit */
        block = NEXT(block);
        block_size = GET_SIZE(HEADER(block));
#ifdef NEXT_FIT
        if (block_size == 0) {    /* reach the epilogue block */
            block = prolog_block; /* go to the prologue block */
        }
        if (block == latest_block) { /* reach the latest touched block */
            block = NULL;
            break;
        }
#else
        if (block_size == 0) {
            block = NULL;
            break;
        }
#endif
    } while (block_size < alloc_size || GET_ALLOC(HEADER(block)));

    dbg_printf("%p = find_fit(0x%lx) is ready to exit.\n", block, alloc_size);
    return block;
}

static void place(void *block, size_t alloc_size)
{
    dbg_printf("place(%p, 0x%lx) is ready to start.\n", block, alloc_size);

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

    dbg_checkheap();
    dbg_printf("place() is ready to exit.\n");
}

/*
 * mm_init - Called when a new trace starts.
 */
int mm_init(void)
{
    dbg_printf("mm_init() is ready to start.\n");

    /* Create the initial empty heap */
    mem_reset_brk();
    assert(mem_heap_lo() == mem_heap_hi() + 1);
    if ((prolog_block = mem_sbrk(4*WSIZE)) == (void *)-1)
        return -1;
    PUT(prolog_block, 0);                          /* Padding on head */
    PUT(prolog_block + (1*WSIZE), PACK(DSIZE, 1)); /* Prologue header */ 
    PUT(prolog_block + (2*WSIZE), PACK(DSIZE, 1)); /* Prologue footer */ 
    PUT(prolog_block + (3*WSIZE), PACK(0, 1));     /* Epilogue header */
    prolog_block += (2*WSIZE);
#ifdef NEXT_FIT
    latest_block = prolog_block;
#endif
    assert(mem_heapsize() == 4*WSIZE);
    dbg_checkheap();

    /* Extend the empty heap with a free block of CHUNK bytes */
    if (extend_heap(CHUNK / WSIZE) == NULL) 
        return -1;

    dbg_checkheap();
    dbg_printf("mm_init() is ready to exit.\n");
    return 0;
}

/*
 * malloc - Allocate a block by ...
 *      ...
 */
void *malloc(size_t size)
{
    dbg_printf("malloc(0x%lx) is ready to start.\n", size);

    size_t alloc_size; /* Adjusted block size */
    size_t chunk_size; /* Amount to extend heap if no fit */
    void *block;

    if (prolog_block == NULL)
        mm_init();

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    if (size <= DSIZE)
        alloc_size = QSIZE; /* payload + overhead */
    else
        alloc_size = DSIZE * (((size + DSIZE) + (DSIZE-1)) / DSIZE);

    /* Search the free list for a fit */
    if ((block = find_fit(alloc_size)) == NULL) {
        /* No fit found. Get more memory. */
        chunk_size = MAX(alloc_size, CHUNK);
        if ((block = extend_heap(chunk_size/WSIZE)) == NULL)  
            return NULL;
    }
    dbg_checkheap();
    place(block, alloc_size);
    dbg_printf("malloc(0x%lx) is ready to exit.\n", size);
#ifdef NEXT_FIT
    latest_block = block;
#endif
    return block;
}

/*
 * free -  ...
 *      ...
 */
void free(void *block)
{
    dbg_printf("free() is ready to start.\n");

    if (block == NULL) 
        return;

    size_t size = GET_SIZE(HEADER(block));
    if (prolog_block == NULL)
        mm_init();

    PUT(HEADER(block), PACK(size, 0));
    PUT(FOOTER(block), PACK(size, 0));
    block = coalesce(block);
#ifdef NEXT_FIT
    latest_block = PREV(block);
#endif
    dbg_printf("free() is ready to exit.\n");
}

/*
 * realloc - Change the size of the block by ...
 *      ...
 */
void *realloc(void *old_block, size_t size)
{
    size_t old_size;
    void *new_block;

    /* If size == 0 then this is just free, and we return NULL. */
    if (size == 0) {
        free(old_block);
        return NULL;
    }

    /* If old_block is NULL, then this is just malloc. */
    if (old_block == NULL) {
        return malloc(size);
    }

    new_block = malloc(size);

    /* If realloc() fails the original block is left untouched  */
    if (!new_block) {
        return NULL;
    }

    /* Copy the old data. */
    old_size = GET_SIZE(HEADER(old_block));
    if (size < old_size)
        old_size = size;
    memcpy(new_block, old_block, old_size);

    /* Free the old block. */
    free(old_block);

    return new_block;
}

/*
 * calloc - Allocate the block and set it to zero.
 */
void *calloc (size_t nmemb, size_t size)
{
    size_t bytes = nmemb * size;
    void *block;

    block = malloc(bytes);
    memset(block, 0, bytes);

    return block;
}

/*
 * mm_checkheap - ...
 *      ...
 */
void mm_checkheap(int verbose){
    void *block = prolog_block;
    int curr_alloc = 1, next_alloc;

    while (GET_SIZE(HEADER(block))) {
        assert(GET(HEADER(block)) == GET(FOOTER(block)));
        block = NEXT(block);
        /* No adjacent free blocks. */
        next_alloc = GET_ALLOC(HEADER(block));
        assert(next_alloc | curr_alloc);
        curr_alloc = next_alloc;
    }
    assert(block == mem_heap_hi() + 1);
    assert(GET(HEADER(block)) == 1);

    if (verbose) {
        dbg_printf("mm_checkheap() succeeds.\n");
    }
}
