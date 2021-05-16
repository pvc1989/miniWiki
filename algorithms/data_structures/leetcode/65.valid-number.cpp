/*
 * @lc app=leetcode id=65 lang=cpp
 *
 * [65] Valid Number
 *
 * https://leetcode.com/problems/valid-number/description/
 *
 * algorithms
 * Hard (15.98%)
 * Likes:    38
 * Dislikes: 90
 * Total Accepted:    205.5K
 * Total Submissions: 1.3M
 * Testcase Example:  '"0"'
 *
 * A valid number can be split up into these components (in order):
 * 
 * 
 * A decimal number or an integer.
 * (Optional) An 'e' or 'E', followed by an integer.
 * 
 * 
 * A decimal number can be split up into these components (in order):
 * 
 * 
 * (Optional) A sign character (either '+' or '-').
 * One of the following formats:
 * 
 * At least one digit, followed by a dot '.'.
 * At least one digit, followed by a dot '.', followed by at least one
 * digit.
 * A dot '.', followed by at least one digit.
 * 
 * 
 * 
 * 
 * An integer can be split up into these components (in order):
 * 
 * 
 * (Optional) A sign character (either '+' or '-').
 * At least one digit.
 * 
 * 
 * For example, all the following are valid numbers: ["2", "0089", "-0.1",
 * "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93",
 * "-123.456e789"], while the following are not valid numbers: ["abc", "1a",
 * "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"].
 * 
 * Given a string s, return true if s is a valid number.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "0"
 * Output: true
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s = "e"
 * Output: false
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: s = "."
 * Output: false
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: s = ".1"
 * Output: true
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 20
 * s consists of only English letters (both uppercase and lowercase), digits
 * (0-9), plus '+', minus '-', or dot '.'.
 * 
 * 
 */

// @lc code=start
class Solution {
  static bool isDigit(char c) {
    return '0' <= c && c <= '9';
  }
 public:
  bool isNumber(string s) {
    int i_dot{-1}, i_exp{-1};
    int n = s.size();
    int i = (s[0] == '+' || s[0] == '-');
    int n_digits = 0, n_digits_all = 0;
    while (i < n) {
      if (isDigit(s[i])) {
        ++n_digits_all;
        ++n_digits;
      }
      else if (s[i] == '.') {
        if (i_dot != -1 || i_exp != -1) {
          return false;
        }
        n_digits = 0;
        i_dot = i;
      }
      else if (s[i] == 'E' || s[i] == 'e') {
        if (i_exp != -1 || n_digits_all == 0) {
          return false;
        }
        n_digits = 0;
        i_exp = i;
      }
      else if (s[i] == '+' || s[i] == '-') {
        if (s[i-1] != 'E' && s[i-1] != 'e') {
          return false;
        }
      }
      else {
        return false;
      }
      ++i;
    }
    if (i_exp != -1) {
      return n_digits;
    }
    if (i_dot != -1) {
      return n_digits_all;
    }
    return true;
  }
};
// @lc code=end

