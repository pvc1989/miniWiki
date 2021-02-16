/*
 * @lc app=leetcode id=1306 lang=cpp
 *
 * [1306] Jump Game III
 *
 * https://leetcode.com/problems/jump-game-iii/description/
 *
 * algorithms
 * Medium (62.56%)
 * Likes:    1006
 * Dislikes: 32
 * Total Accepted:    63.9K
 * Total Submissions: 102.1K
 * Testcase Example:  '[4,2,3,0,3,1,2]\n5'
 *
 * Given an array of non-negative integers arr, you are initially positioned at
 * start index of the array. When you are at index i, you can jump to i +
 * arr[i] or i - arr[i], check if you can reach to any index with value 0.
 * 
 * Notice that you can not jump outside of the array at any time.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: arr = [4,2,3,0,3,1,2], start = 5
 * Output: true
 * Explanation: 
 * All possible ways to reach at index 3 with value 0 are: 
 * index 5 -> index 4 -> index 1 -> index 3 
 * index 5 -> index 6 -> index 4 -> index 1 -> index 3 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: arr = [4,2,3,0,3,1,2], start = 0
 * Output: true 
 * Explanation: 
 * One possible way to reach at index 3 with value 0 is: 
 * index 0 -> index 4 -> index 1 -> index 3
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: arr = [3,0,2,1,2], start = 2
 * Output: false
 * Explanation: There is no way to reach at index 1 with value 0.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= arr.length <= 5 * 10^4
 * 0 <= arr[i] < arr.length
 * 0 <= start < arr.length
 * 
 * 
 */

// @lc code=start
class Solution {
 private:
  vector<bool> touched_;
  vector<int>::size_type size_;
 private:
  bool dfs(vector<int> const &index_to_step, int curr_index) {
    auto step = index_to_step[curr_index];
    if (step == 0) return true;
    for (auto next_index : {curr_index+step, curr_index-step}) {  // O(1)
      if (0 <= next_index && next_index < size_
          && touched_[next_index] == false/* path should be acyclic */) {
        touched_[next_index] = true;
        if (dfs(index_to_step, next_index)) return true;
      }
    }
    return false;
  }
 public:
  bool canReach(vector<int> &index_to_step, int start_index) {
    size_ = index_to_step.size();
    touched_.resize(size_);
    touched_[start_index] = true;
    return dfs(index_to_step, start_index);
  }
};
// @lc code=end

