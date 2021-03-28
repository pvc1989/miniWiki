/*
 * @lc app=leetcode id=647 lang=cpp
 *
 * [647] Palindromic Substrings
 *
 * https://leetcode.com/problems/palindromic-substrings/description/
 *
 * algorithms
 * Medium (61.98%)
 * Likes:    3851
 * Dislikes: 134
 * Total Accepted:    256.4K
 * Total Submissions: 413.5K
 * Testcase Example:  '"abc"'
 *
 * Given a string, your task is to count how many palindromic substrings in
 * this string.
 * 
 * The substrings with different start indexes or end indexes are counted as
 * different substrings even they consist of same characters.
 * 
 * Example 1:
 * 
 * 
 * Input: "abc"
 * Output: 3
 * Explanation: Three palindromic strings: "a", "b", "c".
 * 
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: "aaa"
 * Output: 6
 * Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
 * 
 * 
 * 
 * 
 * Note:
 * 
 * 
 * The input string length won't exceed 1000.
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int countSubstrings(string s) {
    int count = 0;
    int n = s.size();
    if (n == 0) return count;
    // dp[length][head] := s[head, head + length) is palindromic
    auto dp = vector<vector<bool>>(n + 1, vector<bool>(n, true));
    count = n;  // length == 1
    for (int l = 2; l <= n; ++l) {
      for (int h = n - l; h >= 0; --h) {
        if (dp[l - 2][h + 1] && s[h] == s[h + l - 1]) {
          ++count;
        } else {
          dp[l][h] = false;
        }
      }
    }
    return count;
  }
};
// @lc code=end

