/*
 * @lc app=leetcode id=1770 lang=cpp
 *
 * [1770] Maximum Score from Performing Multiplication Operations
 *
 * https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/description/
 *
 * algorithms
 * Medium (28.20%)
 * Likes:    205
 * Dislikes: 74
 * Total Accepted:    6.9K
 * Total Submissions: 24.3K
 * Testcase Example:  '[1,2,3]\n[3,2,1]'
 *
 * You are given two integer arrays nums and multipliers of size n and m
 * respectively, where n >= m. The arrays are 1-indexed.
 * 
 * You begin with a score of 0. You want to perform exactly m operations. On
 * the i^th operation (1-indexed), you will:
 * 
 * 
 * Choose one integer x from either the start or the end of the array nums.
 * Add multipliers[i] * x to your score.
 * Remove x from the array nums.
 * 
 * 
 * Return the maximum score after performing m operations.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [1,2,3], multipliers = [3,2,1]
 * Output: 14
 * Explanation: An optimal solution is as follows:
 * - Choose from the end, [1,2,3], adding 3 * 3 = 9 to the score.
 * - Choose from the end, [1,2], adding 2 * 2 = 4 to the score.
 * - Choose from the end, [1], adding 1 * 1 = 1 to the score.
 * The total score is 9 + 4 + 1 = 14.
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [-5,-3,-3,-2,7,1], multipliers = [-10,-5,3,4,6]
 * Output: 102
 * Explanation: An optimal solution is as follows:
 * - Choose from the start, [-5,-3,-3,-2,7,1], adding -5 * -10 = 50 to the
 * score.
 * - Choose from the start, [-3,-3,-2,7,1], adding -3 * -5 = 15 to the score.
 * - Choose from the start, [-3,-2,7,1], adding -3 * 3 = -9 to the score.
 * - Choose from the end, [-2,7,1], adding 1 * 4 = 4 to the score.
 * - Choose from the end, [-2,7], adding 7 * 6 = 42 to the score. 
 * The total score is 50 + 15 - 9 + 4 + 42 = 102.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * n == nums.length
 * m == multipliers.length
 * 1 <= m <= 10^3
 * m <= n <= 10^5 
 * -1000 <= nums[i], multipliers[i] <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
  int solve(vector<int> const & nums, int head, int last,
            vector<int> const & multipliers, int step,
            vector<vector<int>>* scores) {
    if (step == multipliers.size()) return 0;
    auto score = (*scores)[head][step];
    if (score == INT_MIN) {
      auto multiple = multipliers[step];
      auto cut_head = multiple * nums[head]
          + solve(nums, head + 1, last, multipliers, step + 1, scores);
      auto cut_last = multiple * nums[last]
          + solve(nums, head, last - 1, multipliers, step + 1, scores);
      (*scores)[head][step] = score = max(cut_head, cut_last);
    }
    return score;
  }
 public:
  int maximumScore(vector<int>& nums, vector<int>& multipliers) {
    auto n = nums.size(), m = multipliers.size();
    // dp[s][h] := max score on range `[h, h + n - s)` after `s` steps
    auto dp = vector<vector<int>>(m + 1, vector<int>(m + 1));
    for (int step = m - 1; step >= 0; --step) {
      auto multiple = multipliers[step];
      for (int head = step; head >= 0; --head) {
        auto last = head + n - step - 1;
        dp[step][head] = max(
            multiple * nums[head] + dp[step + 1][head + 1],
            multiple * nums[last] + dp[step + 1][head]);
      }
    }
    return dp[0][0];
  }
};
// @lc code=end

