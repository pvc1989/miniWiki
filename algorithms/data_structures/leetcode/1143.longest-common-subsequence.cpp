/*
 * @lc app=leetcode id=1143 lang=cpp
 *
 * [1143] Longest Common Subsequence
 *
 * https://leetcode.com/problems/longest-common-subsequence/description/
 *
 * algorithms
 * Medium (58.64%)
 * Likes:    2607
 * Dislikes: 32
 * Total Accepted:    181.2K
 * Total Submissions: 308.9K
 * Testcase Example:  '"abcde"\n"ace"'
 *
 * Given two strings text1 and text2, return the length of their longest common
 * subsequence.
 * 
 * A subsequence of a string is a new string generated from the original string
 * with some characters(can be none) deleted without changing the relative
 * order of the remaining characters. (eg, "ace" is a subsequence of "abcde"
 * while "aec" is not). A common subsequence of two strings is a subsequence
 * that is common to both strings.
 * 
 * 
 * 
 * If there is no common subsequence, return 0.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: text1 = "abcde", text2 = "ace" 
 * Output: 3  
 * Explanation: The longest common subsequence is "ace" and its length is 3.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: text1 = "abc", text2 = "abc"
 * Output: 3
 * Explanation: The longest common subsequence is "abc" and its length is 3.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: text1 = "abc", text2 = "def"
 * Output: 0
 * Explanation: There is no such common subsequence, so the result is 0.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= text1.length <= 1000
 * 1 <= text2.length <= 1000
 * The input strings consist of lowercase English characters only.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int longestCommonSubsequence(string x, string y) {
    int m = x.size(), n = y.size();
    auto prev_row = vector<int>(n);
    auto curr_row = vector<int>(n);
    for (int i = 0; i < m; ++i) {
      swap(prev_row, curr_row);
      auto x_i = x[i];
      curr_row[0] = (x_i == y[0]) || prev_row[0];
      for (int j = 1; j < n; ++j) {
          curr_row[j] = (x_i == y[j])
              ? (prev_row[j-1] + 1) : max(prev_row[j], curr_row[j-1]);
      }
    }
    return curr_row.back();
  }
};
// @lc code=end

