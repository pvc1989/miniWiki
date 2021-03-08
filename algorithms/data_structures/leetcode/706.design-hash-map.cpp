/*
 * @lc app=leetcode id=706 lang=cpp
 *
 * [706] Design HashMap
 *
 * https://leetcode.com/problems/design-hashmap/description/
 *
 * algorithms
 * Easy (62.78%)
 * Likes:    1435
 * Dislikes: 156
 * Total Accepted:    162.7K
 * Total Submissions: 254.2K
 * Testcase Example:  '["MyHashMap","put","put","get","get","put","get", "remove", "get"]\n' +
  '[[],[1,1],[2,2],[1],[3],[2,1],[2],[2],[2]]'
 *
 * Design a HashMap without using any built-in hash table libraries.
 * 
 * To be specific, your design should include these functions:
 * 
 * 
 * put(key, value) : Insert a (key, value) pair into the HashMap. If the value
 * already exists in the HashMap, update the value.
 * get(key): Returns the value to which the specified key is mapped, or -1 if
 * this map contains no mapping for the key.
 * remove(key) : Remove the mapping for the value key if this map contains the
 * mapping for the key.
 * 
 * 
 * 
 * Example:
 * 
 * 
 * MyHashMap hashMap = new MyHashMap();
 * hashMap.put(1, 1);          
 * hashMap.put(2, 2);         
 * hashMap.get(1);            // returns 1
 * hashMap.get(3);            // returns -1 (not found)
 * hashMap.put(2, 1);          // update the existing value
 * hashMap.get(2);            // returns 1 
 * hashMap.remove(2);          // remove the mapping for 2
 * hashMap.get(2);            // returns -1 (not found) 
 * 
 * 
 * 
 * Note:
 * 
 * 
 * All keys and values will be in the range of [0, 1000000].
 * The number of operations will be in the range of [1, 10000].
 * Please do not use the built-in HashMap library.
 * 
 * 
 */

// @lc code=start
class MyHashMap {
  static constexpr int M = 10007;
  array<vector<pair<int, int>>, M> chains_;

 public:
  /** Initialize your data structure here. */
  MyHashMap() {
  }

  /** value will always be non-negative. */
  void put(int key, int value) {
    auto& chain = chains_[key % M];
    for (auto& [k, v] : chain) {
      if (k == key) {
        v = value;
        return;
      }
    }
    chain.emplace_back(key, value);
  }

  /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
  int get(int key) {
    auto& chain = chains_[key % M];
    for (auto& [k, v] : chain) {
      if (k == key) {
        return v;
      }
    }
    return -1;
  }

  /** Removes the mapping of the specified value key if this map contains a mapping for the key */
  void remove(int key) {
    auto& chain = chains_[key % M];
    for (auto iter = chain.begin(); iter != chain.end(); ++iter) {
      if (iter->first == key) {
        chain.erase(iter);
        return;
      }
    }
  }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */
// @lc code=end

