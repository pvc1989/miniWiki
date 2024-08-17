#include "lru.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>

MapIter LRU::find(Key const &key) {
  return map_.find(key);
}

void LRU::print() const {
  std::printf("LRU = ");
  for (auto &[key, data] : list_) {
    std::printf("%s(%ld)\n   -> ", key->c_str(), data.size());
  }
  std::printf("NULL\n");
}

void LRU::emplace(Key const &key, char const *data, int size) {
  assert(size <= capacity());
  size_ += size;
  auto [map_iter, _] = map_.emplace(key, list_.end());
  auto list_iter = list_.emplace(list_.end());
  list_iter->key = &(map_iter->first);
  list_iter->data.resize(size);
  std::memcpy(list_iter->data.data(), data, size);
  map_iter->second = list_iter;
  while (this->size() > capacity()) {
    pop();  // evict the LRU item;
  }
}

void LRU::sink(MapIter map_iter) {
  if (map_iter == map_.end()) return;
  auto list_iter = map_iter->second;
  auto &&item = std::move(*list_iter);
  list_.erase(list_iter);
  map_iter->second = list_.emplace(list_.end(), item);
}

void LRU::pop() {
  auto &[key_ptr, data] = list_.front();
  size_ -= data.size();
  map_.erase(*key_ptr);
  list_.pop_front();
}
