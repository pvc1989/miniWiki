/*
 * @lc app=leetcode id=41 lang=cpp
 *
 * [41] First Missing Positive
 *
 * https://leetcode.com/problems/first-missing-positive/description/
 *
 * algorithms
 * Hard (33.61%)
 * Likes:    5224
 * Dislikes: 938
 * Total Accepted:    446.8K
 * Total Submissions: 1.3M
 * Testcase Example:  '[1,2,0]'
 *
 * Given an unsorted integer array nums, find the smallest missing positive
 * integer.
 * 
 * 
 * Example 1:
 * Input: nums = [1,2,0]
 * Output: 3
 * Example 2:
 * Input: nums = [3,4,-1,1]
 * Output: 2
 * Example 3:
 * Input: nums = [7,8,9,11,12]
 * Output: 1
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= nums.length <= 300
 * -2^31 <= nums[i] <= 2^31 - 1
 * 
 * 
 * 
 * Follow up: Could you implement an algorithm that runs in O(n) time and uses
 * constant extra space?
 * 
 */

// @lc code=start
class Solution {
 public:
  int firstMissingPositive(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    auto iter = upper_bound(nums.begin(), nums.end(), 0);
    if (iter == nums.end() || iter[0] != 1) return 1;
    auto last = nums.begin() + nums.size() - 1;
    while (iter != last && iter[1] - iter[0] < 2) { ++iter; }
    return iter[0] + 1;
  }
};
// @lc code=end

