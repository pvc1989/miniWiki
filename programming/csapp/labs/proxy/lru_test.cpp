#include "lru.hpp"
#include <cassert>

int main() {
    char const buf[1024] = "";
    auto lru = LRU(512);
    // test `lru.print` and `lru.emplace`
    lru.print();  // LRU = buf
    lru.emplace("A", buf, 128);
    assert(lru.size() == 128);
    lru.print();  // LRU = A(128) -> buf
    lru.emplace("B", buf, 128);
    assert(lru.size() == 256);
    lru.print();  // LRU = A(128) -> B(128) -> buf
    lru.emplace("C", buf, 256);
    assert(lru.size() == 512);
    lru.print();  // LRU = A(128) -> B(128) -> C(256) -> buf
    // test `lru.find` and `lru.sink`
    lru.sink(lru.find("B"));  // the one in the middle
    lru.print();  // LRU = A(128) -> C(256) -> B(128) -> buf
    lru.sink(lru.find("A"));  // the one at the front
    lru.print();  // LRU = C(256) -> B(128) -> A(128) -> buf
    lru.sink(lru.find("A"));  // the one at the back
    lru.print();  // LRU = C(256) -> B(128) -> A(128) -> buf
    assert(lru.find("C") != lru.end() && lru.find("B") != lru.end()
        && lru.find("A") != lru.end() && lru.find("D") == lru.end());
    lru.print();  // LRU = C(256) -> B(128) -> A(128) -> buf
    // test eviction
    assert(lru.size() == 512);
    lru.emplace("D", buf, 256);  // evict C(256)
    assert(lru.size() == 512);
    lru.print();  // LRU = B(128) -> A(128) -> D(256) -> buf
    lru.emplace("E", buf, 129);  // evict B(128) and A(128)
    assert(lru.size() == 256 + 129);
    lru.print();  // LRU = D(256) -> E(129) -> buf
    lru.emplace("F", buf, 127);  // no eviction
    assert(lru.size() == 512);
    lru.print();  // LRU = D(256) -> E(129) -> F(127) -> buf
    // test `lru.pop`
    lru.pop();
    lru.print();  // LRU = E(256) -> F(127) -> buf
    lru.pop();
    lru.print();  // LRU = F(127) -> buf
    lru.pop();
    lru.print();  // LRU = buf
}
