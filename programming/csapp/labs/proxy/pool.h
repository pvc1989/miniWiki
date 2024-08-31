/**
 * @file pool.h
 * @brief Mimic `pool.h` but using `pthread_mutex_t`.
 * 
 */

#ifndef __POOL__
#define __POOL__

struct _pool;
typedef struct _pool pool_t;

void pool_init(pool_t *p, int n);
void pool_deinit(pool_t *p);
void pool_insert(pool_t *p, int item);
int pool_remove(pool_t *p);

#endif /* __POOL__ */
