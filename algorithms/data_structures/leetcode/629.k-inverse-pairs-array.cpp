/*
 * @lc app=leetcode id=629 lang=cpp
 *
 * [629] K Inverse Pairs Array
 *
 * https://leetcode.com/problems/k-inverse-pairs-array/description/
 *
 * algorithms
 * Hard (31.73%)
 * Likes:    426
 * Dislikes: 80
 * Total Accepted:    13.5K
 * Total Submissions: 41.9K
 * Testcase Example:  '3\n0'
 *
 * For an integer array nums, an inverse pair is a pair of integers [i, j]
 * where 0 <= i < j < nums.length and nums[i] > nums[j].
 * 
 * Given two integers n and k, return the number of different arrays consist of
 * numbers from 1 to n such that there are exactly k inverse pairs. Since the
 * answer can be huge, return it modulo 10^9 + 7.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 3, k = 0
 * Output: 1
 * Explanation: Only the array [1,2,3] which consists of numbers from 1 to 3
 * has exactly 0 inverse pairs.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 3, k = 1
 * Output: 2
 * Explanation: The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 1000
 * 0 <= k <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
  array<array<int64_t, 1001>, 1001> dp_;

  uint64_t dp(int n, int k) {
    auto v = dp_[n][k];
    if (v == -1) {
      if (n*(n-1)/2 < k) {
        v = 0;
      }
      else if (k == 0) {
        v = 1;
      }
      else {
        v = 0;
        for (int i = min(n-1, k); i >= 0; --i) {
          v += dp(n-1, k-i);
        }
      }
      dp_[n][k] = v;
    }
    return v;
  }

 public:
  int kInversePairs(int n, int k) {
    if (n*(n-1)/2 < k)
      return 0;
    for (int m = 1; m <= n; ++m)
      dp_[m][0] = 1;
    for (int m = 2; m <= n; ++m) {
      int j_max = min(m*(m-1)/2, k);
      for (int j = j_max; j > 0; --j) {
        int i = j - min(m-1, j) - 1;
        dp_[m][j] = (dp_[m-1][j] - (i < 0 ? 0 : dp_[m-1][i]) + 1000000007) % 1000000007;
      }
      for (int j = 1; j <= j_max; ++j) {
        // convert to (inclusive) prefix sum
        // to speed up partial sum
        dp_[m][j] += dp_[m][j-1];
        dp_[m][j] %= 1000000007;
      }
    }
    return dp_[n][k];
  }
};
// @lc code=end

