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
 */
struct _lru;
typedef struct _lru lru_t;

lru_t *lru_construct();

void lru_destruct(lru_t *lru);

item_t *lru_find(lru_t *lru, char const *key);

void lru_emplace(lru_t *lru, char const *key, char const *data);

void lru_sink(lru_t *lru, item_t *item);

void lru_pop(lru_t *lru);
