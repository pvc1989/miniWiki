/*
 * @lc app=leetcode id=1664 lang=cpp
 *
 * [1664] Ways to Make a Fair Array
 *
 * https://leetcode.com/problems/ways-to-make-a-fair-array/description/
 *
 * algorithms
 * Medium (61.66%)
 * Likes:    342
 * Dislikes: 7
 * Total Accepted:    11.4K
 * Total Submissions: 18.5K
 * Testcase Example:  '[2,1,6,4]'
 *
 * You are given an integer array nums. You can choose exactly one index
 * (0-indexed) and remove the element. Notice that the index of the elements
 * may change after the removal.
 * 
 * For example, if nums = [6,1,7,4,1]:
 * 
 * 
 * Choosing to remove index 1 results in nums = [6,7,4,1].
 * Choosing to remove index 2 results in nums = [6,1,4,1].
 * Choosing to remove index 4 results in nums = [6,1,7,4].
 * 
 * 
 * An array is fair if the sum of the odd-indexed values equals the sum of the
 * even-indexed values.
 * 
 * Return the number of indices that you could choose such that after the
 * removal, nums is fair. 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [2,1,6,4]
 * Output: 1
 * Explanation:
 * Remove index 0: [1,6,4] -> Even sum: 1 + 4 = 5. Odd sum: 6. Not fair.
 * Remove index 1: [2,6,4] -> Even sum: 2 + 4 = 6. Odd sum: 6. Fair.
 * Remove index 2: [2,1,4] -> Even sum: 2 + 4 = 6. Odd sum: 1. Not fair.
 * Remove index 3: [2,1,6] -> Even sum: 2 + 6 = 8. Odd sum: 1. Not fair.
 * There is 1 index that you can remove to make nums fair.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [1,1,1]
 * Output: 3
 * Explanation: You can remove any index and the remaining array is fair.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: nums = [1,2,3]
 * Output: 0
 * Explanation: You cannot make a fair array after removing any index.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 10^5
 * 1 <= nums[i] <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int waysToMakeFair(vector<int>& a) {
    auto n = a.size();
    auto total_sum = accumulate(a.begin(), a.end(), 0);
    vector<int> prefix__odd_sums(n);  // sum of  odd-indexed values in a[0, i)
    vector<int> suffix_even_sums(n);  // sum of even-indexed values in a(i, n)
    for (int i = 0; i != n - 1; ++i) {
      auto a_i = a[i];
      prefix__odd_sums[i + 1] = prefix__odd_sums[i] + (i % 2 ? a_i : 0);
    }
    for (int i = n - 1; i > 0; --i) {
      auto a_i = a[i];
      suffix_even_sums[i - 1] = suffix_even_sums[i] + (i % 2 ? 0 : a_i);
    }
    int count = 0;
    for (int i = 0; i != n; ++i) {
      auto total__odd_sum = prefix__odd_sums[i] + suffix_even_sums[i];
      auto total_even_sum = total_sum - a[i] - total__odd_sum;
      if (total__odd_sum == total_even_sum) {
        ++count;
      }
    }
    return count;
  }
};
// @lc code=end

