/*
 * @lc app=leetcode id=10 lang=cpp
 *
 * [10] Regular Expression Matching
 *
 * https://leetcode.com/problems/regular-expression-matching/description/
 *
 * algorithms
 * Hard (27.32%)
 * Likes:    5356
 * Dislikes: 824
 * Total Accepted:    509.2K
 * Total Submissions: 1.9M
 * Testcase Example:  '"aa"\n"a"'
 *
 * Given an input string (s) and a pattern (p), implement regular expression
 * matching with support for '.' and '*' where: 
 * 
 * 
 * '.' Matches any single character.​​​​
 * '*' Matches zero or more of the preceding element.
 * 
 * 
 * The matching should cover the entire input string (not partial).
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "aa", p = "a"
 * Output: false
 * Explanation: "a" does not match the entire string "aa".
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s = "aa", p = "a*"
 * Output: true
 * Explanation: '*' means zero or more of the preceding element, 'a'.
 * Therefore, by repeating 'a' once, it becomes "aa".
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: s = "ab", p = ".*"
 * Output: true
 * Explanation: ".*" means "zero or more (*) of any character (.)".
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: s = "aab", p = "c*a*b"
 * Output: true
 * Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore,
 * it matches "aab".
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: s = "mississippi", p = "mis*is*p*."
 * Output: false
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= s.length <= 20
 * 0 <= p.length <= 30
 * s contains only lowercase English letters.
 * p contains only lowercase English letters, '.', and '*'.
 * It is guaranteed for each appearance of the character '*', there will be a
 * previous valid character to match.
 * 
 * 
 */

// @lc code=start
class Solution {
  static bool OneCharMatch(char p, char t) {
    return p != '*' && (p == '.' || p == t);
  }
 public:
  bool isMatch(string text, string pattern) {
    // `dp[i][j] == true` means
    // `pattern[i, pattern.size())` covers `text[j, text.size())`
    auto dp = vector<vector<bool>>(pattern.size()+1);
    for (auto& row : dp) { row.resize(text.size()+1, false); }
    dp.back().back() = true;  // empty pattern only covers empty text
    // build the dp table (bottom-up)
    for (int i = pattern.size()-1; i != -1; --i) {
      for (int j = text.size(); j != -1; --j) {
        if (pattern[i] == '*') {
          dp[i][j] = false;
        } else if (j == text.size()) {
          dp[i][j] = (i+2 <= pattern.size() && dp[i+2][j]
              && pattern[i+1] == '*'/* `` can only be covered by `[a-z]*` */);
        } else if (OneCharMatch(pattern[i], text[j])) {
          // assert(j+1 <= text.size());
          if (i+1 == pattern.size()) {
            dp[i][j] = (j+1 == text.size());
          } else if (pattern[i+1] != '*') {
            dp[i][j] = dp[i+1][j+1];
          } else {
            // assert(i+2 <= pattern.size());
            dp[i][j] = dp[i+2][j] || dp[i][j+1];
          }
        } else if (i+1 == pattern.size()) {
          // assert(false == dp[i][j]);
        } else {
          // assert(i+2 <= pattern.size());
          dp[i][j] = pattern[i+1] == '*' && dp[i+2][j];
        }
      }
    }
    // `dp[0][0] == true` means `pattern` covers `text`
    return dp[0][0];
  }
};
// @lc code=end

