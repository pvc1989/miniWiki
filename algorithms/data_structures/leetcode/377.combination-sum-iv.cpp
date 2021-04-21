/*
 * @lc app=leetcode id=377 lang=cpp
 *
 * [377] Combination Sum IV
 *
 * https://leetcode.com/problems/combination-sum-iv/description/
 *
 * algorithms
 * Medium (46.15%)
 * Likes:    2060
 * Dislikes: 242
 * Total Accepted:    158.4K
 * Total Submissions: 341.1K
 * Testcase Example:  '[1,2,3]\n4'
 *
 * Given an array of distinct integers nums and a target integer target, return
 * the number of possible combinations that add up to target.
 * 
 * The answer is guaranteed to fit in a 32-bit integer.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [1,2,3], target = 4
 * Output: 7
 * Explanation:
 * The possible combination ways are:
 * (1, 1, 1, 1)
 * (1, 1, 2)
 * (1, 2, 1)
 * (1, 3)
 * (2, 1, 1)
 * (2, 2)
 * (3, 1)
 * Note that different sequences are counted as different combinations.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [9], target = 3
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 200
 * 1 <= nums[i] <= 1000
 * All the elements of nums are unique.
 * 1 <= target <= 1000
 * 
 * 
 * 
 * Follow up: What if negative numbers are allowed in the given array? How does
 * it change the problem? What limitation we need to add to the question to
 * allow negative numbers?
 * 
 */

// @lc code=start
class Solution {
  int Count(vector<int> const& nums, int target, array<int, 1001>* counter) {
    int count = counter->at(target);
    if (count == -1) {
      count = 0;
      for (auto i : nums) {
        if (i <= target) {
          count += Count(nums, target - i, counter);
        } else {
          break;
        }
      }
      counter->at(target) = count;
    }
    return count;
  }
 public:
  int combinationSum4(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    auto counter = array<int, 1001>();
    counter.fill(-1);
    counter[0] = 1;
    return Count(nums, target, &counter);
  }
};
// @lc code=end

