/**
 * @file lru.c
 * @author your name (you@domain.com)
 * @brief An implementation of LRU based on linked-list and hash-table.
 * @version 0.1
 * @date 2024-08-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "lru.h"

#include "csapp.h"  // Malloc, Free

#include <assert.h>

// Types:

struct _item {
    char const *key;  // URI from a client
    struct {
      char const *data;  // response from a server
      node_t *node;  // node in a `list`
    } value;
    UT_hash_handle hh;  /* makes this structure hashable */
};

struct _node {
    item_t *item;
    node_t *prev, *next; /* needed for doubly-linked lists */
};

struct _lru {
    item_t *map;
    node_t *list;
};

// Macros:
#define MAP_ERASE(map, item) \
    HASH_DEL(map, item); Free(item)

#define LIST_ERASE(list, node) \
    DL_DELETE(list, node); Free(node)

#define LIST_BACK(list) (list->prev)

// Methods:
lru_t *lru_construct() {
    lru_t *lru = Malloc(sizeof(lru_t));
    lru->map = NULL;
    lru->list = NULL;
    return lru;
}

void lru_destruct(lru_t *lru) {
    if (!lru) return;
    /* Delete each item in the map. */
    // See http://troydhanson.github.io/uthash/userguide.html#_delete_item for details.
    item_t *item, *item_tmp;
    HASH_ITER(hh, lru->map, item, item_tmp) {
        MAP_ERASE(lru->map, item);
    }
    /* Delete each node in the list. */
    // See http://troydhanson.github.io/uthash/utlist.html#_example for details.
    node_t *node, *node_tmp;
    DL_FOREACH_SAFE(lru->list, node, node_tmp) {
        LIST_ERASE(lru->list, node);
    }
    Free(lru);
}

item_t *lru_find(lru_t *lru, char const *key) {
    // See http://troydhanson.github.io/uthash/userguide.html#_find_item for details.
    item_t *item = NULL;
    HASH_FIND_INT(lru->map, &key, item);
    lru_sink(lru, item);
    return item;
}

void lru_emplace(lru_t *lru, char const *key, char const *data) {
    // See http://troydhanson.github.io/uthash/userguide.html#_add_item for details.
    item_t *item = Malloc(sizeof(item_t));
    item->key = key;
    item->value.data = data;
    assert(!lru_find(lru, key));
    HASH_ADD_INT(lru->map, key, item);
    // See http://troydhanson.github.io/uthash/utlist.html#_example for details.
    node_t *node = Malloc(sizeof(node_t));
    node->item = item;
    DL_APPEND(lru->list, node);
    item->value.node = node;
}

void lru_sink(lru_t *lru, item_t *item) {
    if (!item) return;
    node_t *node = item->value.node;
    assert(node);
    if (LIST_BACK(lru->list) == node) {
        /* already at back */
        return;
    }
    if (lru->list == node) {
        /* at front */
        lru->list = node->next;
    }
    DL_DELETE(node->prev, node);
    DL_APPEND(lru->list, node);
}

void lru_pop(lru_t *lru) {
    node_t *old_head = lru->list;
    // pop from the map
    MAP_ERASE(lru->map, old_head->item);
    // pop from the list
    if (lru->list == LIST_BACK(old_head)) {
        // front == back
        Free(old_head);
        lru->list = NULL;
    } else {
        node_t *new_head = old_head->next;
        LIST_ERASE(lru->list, old_head);
        lru->list = new_head;
    }
}

void lru_print(lru_t const *lru) {
    node_t *node;
    printf("LRU = ");
    DL_FOREACH(lru->list, node) printf("%s -> ", node->item->key);
    printf("NULL\n");
}
