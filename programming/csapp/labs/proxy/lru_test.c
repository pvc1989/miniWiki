#include "lru.h"
#include <assert.h>

int main() {
    lru_t *lru = lru_construct(512);
    // test `lru_print` and `lru_emplace`
    lru_print(lru);  // LRU = NULL
    lru_emplace(lru, "A", NULL, 128);
    assert(lru_size(lru) == 128);
    lru_print(lru);  // LRU = A(128) -> NULL
    lru_emplace(lru, "B", NULL, 128);
    assert(lru_size(lru) == 256);
    lru_print(lru);  // LRU = A(128) -> B(128) -> NULL
    lru_emplace(lru, "C", NULL, 256);
    assert(lru_size(lru) == 512);
    lru_print(lru);  // LRU = A(128) -> B(128) -> C(256) -> NULL
    // test `lru_find` and `lru_sink`
    lru_sink(lru, lru_find(lru, "B"));  // the one in the middle
    lru_print(lru);  // LRU = A(128) -> C(256) -> B(128) -> NULL
    lru_sink(lru, lru_find(lru, "A"));  // the one at the front
    lru_print(lru);  // LRU = C(256) -> B(128) -> A(128) -> NULL
    lru_sink(lru, lru_find(lru, "A"));  // the one at the back
    lru_print(lru);  // LRU = C(256) -> B(128) -> A(128) -> NULL
    assert(lru_find(lru, "C") && lru_find(lru, "B")
        && lru_find(lru, "A") && !lru_find(lru, "D"));
    lru_print(lru);  // LRU = C(256) -> B(128) -> A(128) -> NULL
    // test eviction
    assert(lru_size(lru) == 512);
    lru_emplace(lru, "D", NULL, 256);  // evict C(256)
    assert(lru_size(lru) == 512);
    lru_print(lru);  // LRU = B(128) -> A(128) -> D(256) -> NULL
    lru_emplace(lru, "E", NULL, 129);  // evict B(128) and A(128)
    assert(lru_size(lru) == 256 + 129);
    lru_print(lru);  // LRU = D(256) -> E(129) -> NULL
    lru_emplace(lru, "F", NULL, 127);  // no eviction
    assert(lru_size(lru) == 512);
    lru_print(lru);  // LRU = D(256) -> E(129) -> F(127) -> NULL
    // test `lru_pop`
    lru_pop(lru);
    lru_print(lru);  // LRU = E(256) -> F -> NULL
    lru_pop(lru);
    lru_print(lru);  // LRU = F -> NULL
    lru_pop(lru);
    lru_print(lru);  // LRU = NULL
    // clean up
    lru_destruct(lru);
}
