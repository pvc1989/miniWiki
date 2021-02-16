/*
 * @lc app=leetcode id=45 lang=cpp
 *
 * [45] Jump Game II
 *
 * https://leetcode.com/problems/jump-game-ii/description/
 *
 * algorithms
 * Hard (31.37%)
 * Likes:    3639
 * Dislikes: 169
 * Total Accepted:    313.3K
 * Total Submissions: 998.9K
 * Testcase Example:  '[2,3,1,1,4]'
 *
 * Given an array of non-negative integers nums, you are initially positioned
 * at the first index of the array.
 * 
 * Each element in the array represents your maximum jump length at that
 * position.
 * 
 * Your goal is to reach the last index in the minimum number of jumps.
 * 
 * You can assume that you can always reach the last index.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [2,3,1,1,4]
 * Output: 2
 * Explanation: The minimum number of jumps to reach the last index is 2. Jump
 * 1 step from index 0 to 1, then 3 steps to the last index.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [2,3,0,1,4]
 * Output: 2
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 3 * 10^4
 * 0 <= nums[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int jump(vector<int> &max_jump_lengths) {
    auto last_index = max_jump_lengths.size() - 1;
    auto curr_jumps = 0;
    auto max_reachable_index_by_curr_jumps = 0;
    auto max_reachable_index_by_next_jumps = max_jump_lengths[curr_jumps];
    for (auto index = 1; index <= last_index; ++index) {
      if (max_reachable_index_by_curr_jumps < index) {
        ++curr_jumps;
        max_reachable_index_by_curr_jumps = max_reachable_index_by_next_jumps;
      }
      assert(index <= max_reachable_index_by_curr_jumps);
      if (last_index <= max_reachable_index_by_curr_jumps) {
        break;
      }
      max_reachable_index_by_next_jumps = max(
          max_reachable_index_by_next_jumps,
          index + max_jump_lengths[index]
      );
    }
    assert(last_index <= max_reachable_index_by_curr_jumps);
    return curr_jumps;
  }
};
// @lc code=end

