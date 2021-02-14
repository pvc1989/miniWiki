/*
 * @lc app=leetcode id=72 lang=cpp
 *
 * [72] Edit Distance
 *
 * https://leetcode.com/problems/edit-distance/description/
 *
 * algorithms
 * Hard (46.64%)
 * Likes:    5166
 * Dislikes: 66
 * Total Accepted:    334.6K
 * Total Submissions: 717.5K
 * Testcase Example:  '"horse"\n"ros"'
 *
 * Given two strings word1 and word2, return the minimum number of operations
 * required to convert word1 to word2.
 * 
 * You have the following three operations permitted on a word:
 * 
 * 
 * Insert a character
 * Delete a character
 * Replace a character
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: word1 = "horse", word2 = "ros"
 * Output: 3
 * Explanation: 
 * horse -> rorse (replace 'h' with 'r')
 * rorse -> rose (remove 'r')
 * rose -> ros (remove 'e')
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: word1 = "intention", word2 = "execution"
 * Output: 5
 * Explanation: 
 * intention -> inention (remove 't')
 * inention -> enention (replace 'i' with 'e')
 * enention -> exention (replace 'n' with 'x')
 * exention -> exection (replace 'n' with 'c')
 * exection -> execution (insert 'u')
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= word1.length, word2.length <= 500
 * word1 and word2 consist of lowercase English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int minDistance(string x, string y) {
    int m = x.size(), n = y.size();
    if (m == 0) return n;
    if (n == 0) return m;
    auto prev_row = vector<int>(n);
    auto curr_row = vector<int>(n);
    for (int j = n-1; j >= 0; --j) {
      // There are 2 ways to convert `x[m, m)` to `y[j, n)`:
      // 1. insert `n-j` chars into `x`
      // 2. delete `n-j` chars from `y`
      curr_row[j] = n - j;
    }
    for (int i = m-1; i >= 0; --i) {
      swap(prev_row, curr_row);  // prev_row := dp[i+1], curr_row := dp[i]
      auto x_i = x[i];
      auto j = n-1;  // compare `x[i, m)` with `y[n, n)`
        auto value = m - (i+1);  // =: dp[i+1][j+1]
        if (x_i == y[j]) {
          curr_row[j] = value;
        } else {
          // take the minimum of `1 + (dp[i+1][j+1], dp[i+1][j], dp[i][j+1])`
          // but `j+1 == n` is out of range
          curr_row[j] = 1 + min(value, min(prev_row[j], m-i));
        }
      for (j = n-2; j >= 0; --j) {
        // compare `x[i, m)` with `y[j, n)`, all indices are within range
        auto value = prev_row[j+1];
        if (x_i == y[j]) {
          curr_row[j] = value;
        } else {
          // take the minimum of `1 + (dp[i+1][j+1], dp[i+1][j], dp[i][j+1])`
          curr_row[j] = 1 + min(value, min(prev_row[j], curr_row[j+1]));
        }
      }
    }
    return curr_row.front();
  }
};
// @lc code=end

