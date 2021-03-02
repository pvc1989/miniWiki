/*
 * @lc app=leetcode id=416 lang=cpp
 *
 * [416] Partition Equal Subset Sum
 *
 * https://leetcode.com/problems/partition-equal-subset-sum/description/
 *
 * algorithms
 * Medium (44.86%)
 * Likes:    4041
 * Dislikes: 89
 * Total Accepted:    263K
 * Total Submissions: 586.3K
 * Testcase Example:  '[1,5,11,5]'
 *
 * Given a non-empty array nums containing only positive integers, find if the
 * array can be partitioned into two subsets such that the sum of elements in
 * both subsets is equal.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [1,5,11,5]
 * Output: true
 * Explanation: The array can be partitioned as [1, 5, 5] and [11].
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [1,2,3,5]
 * Output: false
 * Explanation: The array cannot be partitioned into equal sum subsets.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 200
 * 1 <= nums[i] <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  bool canPartition(vector<int>& a) {
    auto sum = accumulate(begin(a), end(a), 0);
    if (sum % 2) return false;
    auto target = sum / 2;
    auto n = a.size();
    // `dp[i][s]` := `nums[i, n)` can form a sum of `s`
    auto dp = vector<vector<bool>>(n, vector<bool>(target+1));
    for (auto &row : dp) {
      row[0] = true;
    }
    if (a.back() <= target) {
      dp.back()[a.back()] = true;
    }
    for (int i = n - 2; i >= 0; --i) {
      for (int s = 1; s <= target; ++s) {
        if (a[i] <= s) {
          dp[i][s] = dp[i+1][s] || dp[i+1][s-a[i]];
        } else {
          dp[i][s] = dp[i+1][s];
        }
        if (dp[i][target]) return true;
      }
    }
    return dp[0][target];
  }
};
// @lc code=end

