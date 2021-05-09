/*
 * @lc app=leetcode id=665 lang=cpp
 *
 * [665] Non-decreasing Array
 *
 * https://leetcode.com/problems/non-decreasing-array/description/
 *
 * algorithms
 * Medium (19.82%)
 * Likes:    2669
 * Dislikes: 609
 * Total Accepted:    130K
 * Total Submissions: 649.1K
 * Testcase Example:  '[4,2,3]'
 *
 * Given an array nums with n integers, your task is to check if it could
 * become non-decreasing by modifying at most one element.
 * 
 * We define an array is non-decreasing if nums[i] <= nums[i + 1] holds for
 * every i (0-based) such that (0 <= i <= n - 2).
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [4,2,3]
 * Output: true
 * Explanation: You could modify the first 4 to 1 to get a non-decreasing
 * array.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [4,2,1]
 * Output: false
 * Explanation: You can't get a non-decreasing array by modify at most one
 * element.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * n == nums.length
 * 1 <= n <= 10^4
 * -10^5 <= nums[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  bool checkPossibility(vector<int>& nums) {
    int decr_count = 0;
    int decr_i = -1;
    for (int i = nums.size() - 2; i >= 0; --i) {
      if (nums[i] > nums[i+1]) {
        if (++decr_count > 1)
          return false;
        decr_i = i;
      }
    }
    return decr_count == 0
        || decr_i == 0/* decrease nums[0] */
        || decr_i == nums.size() - 2/* increase nums[nums.size() - 1] */
        || nums[decr_i+0] <= nums[decr_i+2]/* increase nums[decr_i+1] */
        || nums[decr_i-1] <= nums[decr_i+1]/* decrease nums[decr_i+0] */;
  }
};
// @lc code=end

