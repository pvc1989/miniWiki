/*
 * @lc app=leetcode id=1345 lang=cpp
 *
 * [1345] Jump Game IV
 *
 * https://leetcode.com/problems/jump-game-iv/description/
 *
 * algorithms
 * Hard (41.94%)
 * Likes:    554
 * Dislikes: 41
 * Total Accepted:    29.9K
 * Total Submissions: 71.4K
 * Testcase Example:  '[100,-23,-23,404,100,23,23,23,3,404]'
 *
 * Given an array of integers arr, you are initially positioned at the first
 * index of the array.
 * 
 * In one step you can jump from index i to index:
 * 
 * 
 * i + 1 where: i + 1 < arr.length.
 * i - 1 where: i - 1 >= 0.
 * j where: arr[i] == arr[j] and i != j.
 * 
 * 
 * Return the minimum number of steps to reach the last index of the array.
 * 
 * Notice that you can not jump outside of the array at any time.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: arr = [100,-23,-23,404,100,23,23,23,3,404]
 * Output: 3
 * Explanation: You need three jumps from index 0 --> 4 --> 3 --> 9. Note that
 * index 9 is the last index of the array.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: arr = [7]
 * Output: 0
 * Explanation: Start index is the last index. You don't need to jump.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: arr = [7,6,9,6,9,6,9,7]
 * Output: 1
 * Explanation: You can jump directly from index 0 to index 7 which is last
 * index of the array.
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: arr = [6,1,9]
 * Output: 2
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: arr = [11,22,7,7,7,7,7,7,7,22,13]
 * Output: 3
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= arr.length <= 5 * 10^4
 * -10^8 <= arr[i] <= 10^8
 * 
 * 
 */

// @lc code=start
class Solution {
  unordered_map<int, unordered_set<int>> value_to_index_set_;
  vector<int> index_to_level_;
  queue<int> index_queue_;
  int size_;

  void enqueue(int index, int level) {
    if (0 <= index && index < size_/* enqueue valid index only */
        && index_to_level_[index] < 0/* enqueue new index only */) {
      index_to_level_[index] = level;
      index_queue_.emplace(index);
    }
  }

 public:
  int minJumps(vector<int> &index_to_value) {
    // initialization
    size_ = index_to_value.size();
    for (int index = 0; index != size_; ++index) {  // O(N)
      auto value = index_to_value[index];
      value_to_index_set_[value].emplace(index);
    }
    index_to_level_.resize(size_, -1);
    enqueue(/* index = */0, /* level = */0);
    // breadth-first search
    while (!index_queue_.empty()) {
      auto index = index_queue_.front(); index_queue_.pop();
      auto next_level = index_to_level_[index] + 1;
      assert(0 < next_level);
      // enqueue its "index" neighbors
      enqueue(index+1, next_level);
      enqueue(index-1, next_level);
      // enqueue its "value" neighbors
      auto value = index_to_value[index];
      auto &index_set = value_to_index_set_[value];
      if (!index_set.empty()) {
        // The loop body will be skipped, except in the first run.
        for (auto another_index : index_set) {
          enqueue(another_index, next_level);
        }
        index_set.clear();
      }
    }
    return index_to_level_.back();
  }
};
// @lc code=end

