#include "csapp.h"
#include "pool.h"

#include <pthread.h>

// For non-static initialization.
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t *_mutex;  // should be guarded by a mutex, too
void mutex_init() {
    pthread_mutex_init(_mutex, NULL);
}

struct _pool {
    /* a FIFO queue */
    int *buf;          /* Buffer array */
    int n;             /* Maximum number of slots */
    int front;         /* buf[(front+1)%n] is first item */
    int rear;          /* buf[rear%n] is last item */
    /* queue and counting semaphores */
    pthread_mutex_t queue;  /* Protects accesses to the queue. */
    sem_t slots;  /* Counts available slots. */
    sem_t items;  /* Counts available items. */
};

/* Create an empty, bounded, shared FIFO buffer with n slots */
void pool_init(pool_t *p, int n) {
    p->buf = Calloc(n, sizeof(int));
    p->n = n;                   /* Buffer holds max of n items */
    p->front = p->rear = 0;     /* Empty buffer iff front == rear */
    _mutex = &p->queue;
    pthread_once(&once, mutex_init);
    Sem_init(&p->slots, 0, n);  /* Initially, buf has n empty slots */
    Sem_init(&p->items, 0, 0);  /* Initially, buf has zero data items */
}

/* Clean up buffer p */
void pool_deinit(pool_t *p) {
    Free(p->buf);
    pthread_mutex_destroy(&p->queue);
}

/* Insert item onto the rear of shared buffer p */
void pool_insert(pool_t *p, int item) {
    P(&p->slots);                       /* Wait for available slot */
    pthread_mutex_lock(&p->queue);      /* Lock the buffer */
    p->buf[(++p->rear)%(p->n)] = item;  /* Insert the item */
    pthread_mutex_unlock(&p->queue);    /* Unlock the buffer */
    V(&p->items);                       /* Announce available item */
}

/* Remove and return the first item from buffer p */
int pool_remove(pool_t *p) {
    int item;
    P(&p->items);                        /* Wait for available item */
    pthread_mutex_lock(&p->queue);       /* Lock the buffer */
    item = p->buf[(++p->front)%(p->n)];  /* Remove the item */
    pthread_mutex_unlock(&p->queue);     /* Unlock the buffer */
    V(&p->slots);                        /* Announce available slot */
    return item;
}
