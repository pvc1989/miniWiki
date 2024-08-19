#include <string>
#include <unordered_map>
#include <list>

using Key = std::string;
using Data = std::string;

struct Item {
  Key const *key;
  Data data;
};

using List = std::list<Item>;
using ListIter = typename List::iterator;

using Map = std::unordered_map<Key, ListIter>;
using MapIter = typename Map::iterator;
using MapIterConst = typename Map::const_iterator;

class LRU {
  Map map_;
  List list_;
  int capacity_, size_ = 0;

 public:
  explicit LRU(int capacity)
      : capacity_(capacity) {}

  ~LRU() = default;

  int capacity() const {
    return capacity_;
  }

  int size() const {
    return size_;
  }

  MapIterConst end() const {
    return map_.end();
  }

  MapIterConst find(Key const &key) const;
  MapIter find(Key const &key);

  bool consistent() const;

  void print() const;

  void emplace(Key const &key, char const *data, int size);
  void sink(MapIter iter);
  void pop();
};
