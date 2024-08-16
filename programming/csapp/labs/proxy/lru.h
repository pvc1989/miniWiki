/**
 * @file lru.h
 * @author your name (you@domain.com)
 * @brief The interface of LRU based on linked-list and hash-table.
 * @version 0.1
 * @date 2024-08-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "uthash.h"  // mimic `std::unordered_map`
#include "utlist.h"  // mimic `std::list`

/**
 * @brief node of a `list`
 * 
 */
struct _node;
typedef struct _node node_t;

/**
 * @brief item of an `unordered_map`
 * 
 */
struct _item;
typedef struct _item item_t;

/**
 * @brief LRU object based on `list` and `unordered_map`
 * 
 * It keeps a list, in which the LRU (least-recently-used) node is at the front, while the the MRU (most-recently-used) node is at the back.
 * 
 */
struct _lru;
typedef struct _lru lru_t;

lru_t *lru_construct(int capacity);

void lru_destruct(lru_t *lru);

int lru_size(lru_t const *lru);

char const *item_data(item_t const *item);

int item_size(item_t const *item);

item_t *lru_find(lru_t const *lru, char const *key);

void lru_emplace(lru_t *lru, char const *key, char const *data, int size);

void lru_sink(lru_t *lru, item_t *item);

void lru_pop(lru_t *lru);

void lru_print(lru_t const *lru);
