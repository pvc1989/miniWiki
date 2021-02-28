/*
 * @lc app=leetcode id=125 lang=cpp
 *
 * [125] Valid Palindrome
 *
 * https://leetcode.com/problems/valid-palindrome/description/
 *
 * algorithms
 * Easy (38.23%)
 * Likes:    1780
 * Dislikes: 3612
 * Total Accepted:    800.2K
 * Total Submissions: 2.1M
 * Testcase Example:  '"A man, a plan, a canal: Panama"'
 *
 * Given a string s, determine if it is a palindrome, considering only
 * alphanumeric characters and ignoring cases.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "A man, a plan, a canal: Panama"
 * Output: true
 * Explanation: "amanaplanacanalpanama" is a palindrome.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s = "race a car"
 * Output: false
 * Explanation: "raceacar" is not a palindrome.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 2 * 10^5
 * s consists only of printable ASCII characters.
 * 
 * 
 */

// @lc code=start
class Solution {
 static bool is_alphanumeric(char c) {
   return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') ||
          ('0' <= c && c <= '9');
 }
 public:
  bool isPalindrome(string s) {
    for (auto& c : s) {
      if ('A' <= c && c <= 'Z') {
        constexpr int d = 'a' - 'A';
        c += d;
      }
    }
    int i = -1, j = s.size(), n = s.size();
    while (i < j) {
      ++i; --j;
      while (i != n && !is_alphanumeric(s[i])) {
        ++i;
      }
      while (0 <= j && !is_alphanumeric(s[j])) {
        --j;
      }
      if (i != n && 0 <= j && s[i] != s[j]) {
        return false;
      }
    }
    return true;
  }
};
// @lc code=end

