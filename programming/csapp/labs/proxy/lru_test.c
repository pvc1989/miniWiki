#include "lru.h"
#include <assert.h>

int main() {
    lru_t *lru = lru_construct();
    // test `lru_print` and `lru_emplace`
    lru_print(lru);  // LRU = NULL
    lru_emplace(lru, "A", NULL);
    lru_print(lru);  // LRU = A -> NULL
    lru_emplace(lru, "B", NULL);
    lru_print(lru);  // LRU = A -> B -> NULL
    lru_emplace(lru, "C", NULL);
    lru_print(lru);  // LRU = A -> B -> C -> NULL
    // test `lru_find` and `lru_sink` (implicitly)
    lru_find(lru, "B");  // the one in the middle
    lru_print(lru);  // LRU = A -> C -> B -> NULL
    lru_find(lru, "A");  // the one at the front
    lru_print(lru);  // LRU = C -> B -> A -> NULL
    lru_find(lru, "A");  // the one at the back
    lru_print(lru);  // LRU = C -> B -> A -> NULL
    assert(lru_find(lru, "C") && lru_find(lru, "B")
        && lru_find(lru, "A") && !lru_find(lru, "D"));
    lru_print(lru);  // LRU = C -> B -> A -> NULL
    // test `lru_pop`
    lru_pop(lru);
    lru_print(lru);  // LRU = B -> A -> NULL
    lru_pop(lru);
    lru_print(lru);  // LRU = A -> NULL
    lru_pop(lru);
    lru_print(lru);  // LRU = NULL
    // clean up
    lru_destruct(lru);
}
