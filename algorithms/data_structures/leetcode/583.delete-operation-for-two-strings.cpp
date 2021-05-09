/*
 * @lc app=leetcode id=583 lang=cpp
 *
 * [583] Delete Operation for Two Strings
 *
 * https://leetcode.com/problems/delete-operation-for-two-strings/description/
 *
 * algorithms
 * Medium (50.25%)
 * Likes:    1545
 * Dislikes: 37
 * Total Accepted:    66.3K
 * Total Submissions: 130.5K
 * Testcase Example:  '"sea"\n"eat"'
 *
 * Given two strings word1 and word2, return the minimum number of steps
 * required to make word1 and word2 the same.
 * 
 * In one step, you can delete exactly one character in either string.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: word1 = "sea", word2 = "eat"
 * Output: 2
 * Explanation: You need one step to make "sea" to "ea" and another step to
 * make "eat" to "ea".
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: word1 = "leetcode", word2 = "etco"
 * Output: 4
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= word1.length, word2.length <= 500
 * word1 and word2 consist of only lowercase English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int minDistance(string word1, string word2) {
    int n1 = word1.size(), n2 = word2.size();
    auto dp = vector<vector<int>>(n1 + 1);
    for (auto& row : dp)
      row.resize(n2 + 1);
    for (int i1 = n1 - 1; i1 >= 0; --i1)
      dp[i1][n2] = n1 - i1;
    for (int i2 = n2 - 1; i2 >= 0; --i2)
      dp[n1][i2] = n2 - i2;
    for (int i1 = n1 - 1; i1 >= 0; --i1) {
      for (int i2 = n2 - 1; i2 >= 0; --i2) {
        if (word1[i1] == word2[i2])
          dp[i1][i2] = dp[i1 + 1][i2 + 1];
        else
          dp[i1][i2] = 1 + min(dp[i1][i2 + 1], dp[i1 + 1][i2]);
      }
    }
    return dp[0][0];
  }
};
// @lc code=end

