/*
 * @lc app=leetcode id=46 lang=cpp
 *
 * [46] Permutations
 *
 * https://leetcode.com/problems/permutations/description/
 *
 * algorithms
 * Medium (66.44%)
 * Likes:    5429
 * Dislikes: 126
 * Total Accepted:    760.1K
 * Total Submissions: 1.1M
 * Testcase Example:  '[1,2,3]'
 *
 * Given an array nums of distinct integers, return all the possible
 * permutations. You can return the answer in any order.
 * 
 * 
 * Example 1:
 * Input: nums = [1,2,3]
 * Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 * Example 2:
 * Input: nums = [0,1]
 * Output: [[0,1],[1,0]]
 * Example 3:
 * Input: nums = [1]
 * Output: [[1]]
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 6
 * -10 <= nums[i] <= 10
 * All the integers of nums are unique.
 * 
 * 
 */

// @lc code=start
class Solution {
  void permute(vector<int> nums, int head, vector<vector<int>> *permutations) {
    if (head == nums.size()) {
      permutations->emplace_back(move(nums));
      return;
    }
    for (int curr = head; curr != nums.size(); ++curr) {
      swap(nums[head], nums[curr]);
      permute(nums, head+1, permutations);
      swap(nums[curr], nums[head]);
    }
  }
 public:
  vector<vector<int>> permute(vector<int> &nums) {
    auto permutations = vector<vector<int>>();
    permute(nums, 0, &permutations);
    return permutations;
  }
};
// @lc code=end

