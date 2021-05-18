/*
 * mm_explicit.c - ...
 *
 * In this approach, a block is allocated by ...
 *
 * This code is faster than ...
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
#ifndef NDEBUG
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

#define WORD_1X     4                /* single word size (bytes) */
#define WORD_2X     8                /* double word size (bytes) */
#define WORD_4X     16               /*  quad  word size (bytes) */
#define MIN_BLOCK_SIZE (WORD_4X + WORD_2X)     /* payload + tags */
#define PAGE       (1<<12) /* extend heap by this amount (bytes) */

/* single word (4) or double word (8) alignment */
#define ALIGNMENT WORD_2X

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT-1)) & ~0x7)

/* PACK a size and allocated bit into a word */
#define PACK(size, alloc)  ((size) | (alloc))

/* GET and PUT a word at address p */
#define GET_WORD(p)        (*(unsigned int *)(p)         )
#define PUT_WORD(p, word)  (*(unsigned int *)(p) = (word))

/* GET the SIZE and ALLOCated fields from address p */
#define GET_SIZE(p)   (GET_WORD(p) & ~0x7)
#define GET_ALOC(p)  (GET_WORD(p) &  0x1)

/* Given block (ptr), get the address of its HEADER and FOOTER */
#define HEADER(block)  ((char *)(block) - WORD_1X)
#define FOOTER(block)  ((char *)(block) + GET_SIZE(HEADER(block)) - WORD_2X)

#define IS_ALOC(block)    (GET_ALOC(HEADER(block)))
#define IS_FREE(block)    (!IS_ALOC(block))

/* block to its NEXT and PREVious blocks in Virtual Memory */
#define VM_NEXT(block) ((char *)(block) + GET_SIZE(HEADER(block)))
#define VM_PREV(block) ((char *)(block) - GET_SIZE(((char *)(block) - WORD_2X)))

/* block to its NEXT and PREVious blocks in Free List */
#define FL_NEXT(block) (*((void **)(block)    ))
#define FL_PREV(block) (*((void **)(block) + 1))

/* Global variables */
static char *first_free_block = NULL; /* Pointer to the first free block */
#ifndef NDEBUG
int free_block_count = 0;
int aloc_block_count = 0;
#define INCR(val) (++(val))
#define DECR(val) (--(val))
#define SET(x, n) ((x) = (n))
#define GET(x)    ((x))
#else
#define INCR(val)
#define DECR(val)
#define SET(x, n)
#define GET(x)    (0)
#endif

/* Private methods */

/*
 * Insert the new block at the head of the list.
 */
static void put_before_head(void *block)
{
    assert(block != first_free_block);
    assert(IS_FREE(block));
    FL_NEXT(block) = first_free_block;
    if (first_free_block) {
        assert(FL_PREV(first_free_block) == NULL);
        FL_PREV(first_free_block) = block;
    }
    first_free_block = block;
    FL_PREV(first_free_block) = NULL;
}

static inline void link_together(void *fl_prev, void* fl_next) {
    if (fl_prev)
        FL_NEXT(fl_prev) = fl_next;
    else
        first_free_block = fl_next;

    if (fl_next)
        FL_PREV(fl_next) = fl_prev;
}

static void *coalesce(void *block)
{
    dbg_printf("coalesce(%p) is ready to start.\n", block);

    char *vm_prev = VM_PREV(block);
    char *fl_prev_of_vm_prev;
    char *fl_next_of_vm_prev;
    char *vm_next = VM_NEXT(block);
    char *fl_prev_of_vm_next;
    char *fl_next_of_vm_next;

    int prev_free = IS_FREE(vm_prev);
    int next_free = IS_FREE(vm_next);
    size_t size = GET_SIZE(HEADER(block));
    if (prev_free) {
      size += GET_SIZE(HEADER(vm_prev));
      DECR(free_block_count);
    }
    if (next_free) {
      size += GET_SIZE(HEADER(vm_next));
      DECR(free_block_count);
    }

    if (prev_free && next_free) {            /* Case 4 */
        fl_prev_of_vm_prev = FL_PREV(vm_prev);
        fl_next_of_vm_prev = FL_NEXT(vm_prev);
        fl_prev_of_vm_next = FL_PREV(vm_next);
        fl_next_of_vm_next = FL_NEXT(vm_next);
        block = vm_prev;
        if (fl_next_of_vm_prev == vm_next) {
            assert(fl_prev_of_vm_next == vm_prev);
            link_together(fl_prev_of_vm_prev, fl_next_of_vm_next);
        }
        else if (fl_prev_of_vm_prev == vm_next) {
            assert(fl_next_of_vm_next == vm_prev);
            link_together(fl_prev_of_vm_next, fl_next_of_vm_prev);
        }
        else {
            assert(fl_prev_of_vm_next != vm_prev);
            assert(fl_next_of_vm_next != vm_prev);
            link_together(fl_prev_of_vm_prev, fl_next_of_vm_prev);
            link_together(fl_prev_of_vm_next, fl_next_of_vm_next);
        }
    }
    else if (prev_free && !next_free) {      /* Case 3 */
        fl_prev_of_vm_prev = FL_PREV(vm_prev);
        fl_next_of_vm_prev = FL_NEXT(vm_prev);
        link_together(fl_prev_of_vm_prev, fl_next_of_vm_prev);
        block = vm_prev;
    }
    else if (!prev_free && next_free) {      /* Case 2 */
        fl_prev_of_vm_next = FL_PREV(vm_next);
        fl_next_of_vm_next = FL_NEXT(vm_next);
        link_together(fl_prev_of_vm_next, fl_next_of_vm_next);
    }
    else {                                   /* Case 1 */
    }
    PUT_WORD(HEADER(block), PACK(size, 0));
    PUT_WORD(FOOTER(block), PACK(size, 0));
    put_before_head(block);

    dbg_printf("coalesce() -> %p is ready to exit.\n", block);
    return block;
}
 
static void *extend_heap(size_t size)
{
    dbg_printf("extend_heap(%ld=0x%lx) is ready to start.\n", size, size);

    char *block;

    block = mem_sbrk(size);
    if (block == (void *)-1)
        return NULL;

    /* Initialize free block header/footer and the epilogue header */
    PUT_WORD(HEADER(block), PACK(size, 0));         /* Free block header */
    PUT_WORD(FOOTER(block), PACK(size, 0));         /* Free block footer */
    PUT_WORD(HEADER(VM_NEXT(block)), PACK(0, 1)); /* New epilogue header */

    INCR(free_block_count);
    block = coalesce(block);

    dbg_printf("extend_heap(%ld=0x%lx) -> %p is ready to exit.\n", size, size, block);
    return block;
}

static void *find_fit(size_t alloc_size)
{
    dbg_printf("find_fit(%ld=0x%lx) is ready to start.\n", alloc_size, alloc_size);

    char *block = first_free_block;
    size_t size; /* size of current block */

#ifdef FIRST_FIT
    while (block && (size = GET_SIZE(HEADER(block))) < alloc_size)
        block = FL_NEXT(block);
#elif defined BEST_FIT
    size_t best_size = alloc_size + WORD_2X + WORD_4X;
    char *best_block = NULL;
    while (block) {
        size = GET_SIZE(HEADER(block));
        if (size == alloc_size) {
            best_block = block;
            break;
        }
        else if (alloc_size < size && (size <= best_size || !best_block)) {
            best_size = size;
            best_block = block;
        }
        else {
            assert(size < alloc_size || (best_size < size && best_block));
        }
        block = FL_NEXT(block);
    }
    block = best_block;
#elif defined GOOD_FIT
    size_t alloc_size_x2 = alloc_size * 2;
    size_t good_size = 0;
    char *good_block = NULL;
    while (block) {
        size = GET_SIZE(HEADER(block));
        if (alloc_size <= size && size <= alloc_size_x2) {
            good_block = block;
            break;
        }
        else if (alloc_size_x2 < size && (size < good_size || !good_block)) {
            good_size = size;
            good_block = block;
        }
        else {
            assert(size < alloc_size || (alloc_size_x2 < size && good_block));
        }
        block = FL_NEXT(block);
    }
    block = good_block;
#endif
    dbg_printf("find_fit(%ld=0x%lx) -> %p is ready to exit.\n", alloc_size, alloc_size, block);
    return block;
}

static void place(void *block, size_t alloc_size)
{
    dbg_printf("place(%p, %ld=0x%lx) is ready to start.\n", block, alloc_size, alloc_size);

    assert(IS_FREE(block));
    char *fl_prev = FL_PREV(block);
    char *fl_next = FL_NEXT(block);
    size_t block_size = GET_SIZE(HEADER(block));
    int split = (block_size >= alloc_size + MIN_BLOCK_SIZE);

    if (split) {
        /* The remaining of current block can hold a min-sized block. */
        PUT_WORD(HEADER(block), PACK(alloc_size, 1));
        PUT_WORD(FOOTER(block), PACK(alloc_size, 1));
        block = VM_NEXT(block);
        block_size -= alloc_size;
    }
    else {
        DECR(free_block_count);
    }
    INCR(aloc_block_count);
    PUT_WORD(HEADER(block), PACK(block_size, !split));
    PUT_WORD(FOOTER(block), PACK(block_size, !split));

    /* fix links */
    link_together(fl_prev, fl_next);
    if (split)
        put_before_head(block);

    dbg_checkheap();
    dbg_printf("place(%p) is ready to exit.\n", block - (split ? alloc_size : 0));
}

/*
 * mm_init - Called when a new trace starts.
 */
int mm_init(void)
{
    dbg_printf("mm_init() is ready to start.\n");

    char* block;
    /* Create the initial empty heap */
    mem_reset_brk();
    assert(mem_heap_lo() == mem_heap_hi() + 1);

    /* Extend the empty heap with a free block of PAGE bytes */
    assert(WORD_1X <= ALIGNMENT && ALIGNMENT <= WORD_4X);
    block = mem_sbrk(WORD_4X);
    assert(block == mem_heap_lo());
    PUT_WORD(block, 0);                                /* Padding on head */
    PUT_WORD(block + (WORD_1X), PACK(WORD_2X, 1));     /* Prologue header */
    PUT_WORD(block + (WORD_2X), PACK(WORD_2X, 1));     /* Prologue footer */
    PUT_WORD(block + (WORD_2X + WORD_1X), PACK(0, 1)); /* Epilogue header */
    SET(free_block_count, 0);
    SET(aloc_block_count, 2);

    first_free_block = NULL;
    block = extend_heap(PAGE - WORD_4X);
    if (block == NULL)
        return -1;
    assert(block == first_free_block);
    assert(block == mem_heap_lo() + WORD_4X);
    dbg_checkheap();

    dbg_printf("mm_init() is ready to exit.\n\n");
    return 0;
}

/*
 * malloc - Allocate a block by ...
 *      ...
 */
void *malloc(size_t size)
{
    dbg_printf("malloc(%ld=0x%lx) is ready to start.\n", size, size);

    void *block;

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    size = ALIGN(MAX(size, WORD_4X)/* payload */ + WORD_2X/* overhead */);

    /* Search the free list for a fit */
    if ((block = find_fit(size)) == NULL) {
        /* No fit found. Get more memory. */
        block = extend_heap(((size + PAGE - 1) / PAGE) * PAGE);
        if (block == NULL)  
            return NULL;
    }
    place(block, size);
    dbg_printf("malloc(%ld=0x%lx (aligned)) -> %p is ready to exit.\n\n", size, size, block);
    return block;
}

/*
 * free -  ...
 *      ...
 */
void free(void *block)
{
    dbg_printf("free(%p) is ready to start.\n", block);

    if (block == NULL)
        return;

    dbg_printf("GET_WORD(HEADER(%p)) == 0x%x\n", block, *((unsigned int*)block-1));
    assert(IS_ALOC(block));
    size_t size = GET_SIZE(HEADER(block));
    PUT_WORD(HEADER(block), PACK(size, 0));
    PUT_WORD(FOOTER(block), PACK(size, 0));
    DECR(aloc_block_count);
    INCR(free_block_count);
    block = coalesce(block);

    dbg_printf("     0x%lx bytes have been freed.\n", size);
    dbg_printf("free(%p) is ready to exit.\n", block);
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
void mm_checkheap(int verbose)
{
    int n_free_blocks = 0, n_aloc_blocks = 0;
    char *fl_prev, *fl_next;

    /* start from prologue block */
    char *block = mem_heap_lo() + ALIGNMENT;
    char *header = HEADER(block);
    size_t size = GET_SIZE(header);

    while (size > 0) {
        /* check each block */
        assert((size_t)mem_heap_lo() <= (size_t)header);
        assert((size_t)FOOTER(block) <= (size_t)mem_heap_hi());
        assert((size_t)block % ALIGNMENT == 0); /* alignment */
        assert(GET_WORD(header) == GET_WORD(FOOTER(block))); /* tag matching */
        assert(MIN_BLOCK_SIZE <= size || size == WORD_2X);
        if (IS_FREE(block)) {
            ++n_free_blocks;
            fl_prev = FL_PREV(block);
            if (fl_prev) {
                assert(block == FL_NEXT(fl_prev));
                assert(IS_FREE(fl_prev));
            }
            else {
                /* `block` is the first free block */
                assert(block == first_free_block);
            }
            fl_next = FL_NEXT(block);
            if (fl_next) {
                assert(block == FL_PREV(fl_next));
                assert(IS_FREE(fl_next));
            }
            else {
                /* `block` is the last free block */
            }
        }
        else {
            ++n_aloc_blocks;
        }
        block = VM_NEXT(block);
        header = HEADER(block);
        size = GET_SIZE(header);
    }
    ++n_aloc_blocks; /* epilogue block is allocated */
    assert(n_aloc_blocks == GET(aloc_block_count));
    assert(n_free_blocks == GET(free_block_count));

    /* traverse the free list */
    n_free_blocks = 0;
    block = first_free_block;
    while (block) {
        /* no consecutive free blocks */
        assert(IS_ALOC(VM_PREV(block)));
        assert(IS_FREE(block));
        assert(IS_ALOC(VM_NEXT(block)));
        /* update counter and pointer */
        ++n_free_blocks;
        block = FL_NEXT(block);
    }
    assert(n_free_blocks == GET(free_block_count));

    if (verbose) {
        dbg_printf("mm_checkheap() succeeds.\n");
    }
}
