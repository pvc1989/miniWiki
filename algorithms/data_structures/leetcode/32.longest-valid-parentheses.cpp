/*
 * @lc app=leetcode id=32 lang=cpp
 *
 * [32] Longest Valid Parentheses
 *
 * https://leetcode.com/problems/longest-valid-parentheses/description/
 *
 * algorithms
 * Hard (29.38%)
 * Likes:    4973
 * Dislikes: 182
 * Total Accepted:    364.7K
 * Total Submissions: 1.2M
 * Testcase Example:  '"(()"'
 *
 * Given a string containing just the characters '(' and ')', find the length
 * of the longest valid (well-formed) parentheses substring.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "(()"
 * Output: 2
 * Explanation: The longest valid parentheses substring is "()".
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s = ")()())"
 * Output: 4
 * Explanation: The longest valid parentheses substring is "()()".
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: s = ""
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= s.length <= 3 * 10^4
 * s[i] is '(', or ')'.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int longestValidParentheses(string s) {
    int length = 0;
    auto lefts = stack<int>();
    auto r_to_l = vector<int>(s.size() + 1);
    iota(r_to_l.begin(), r_to_l.end(), 0);
    for (int i = 0; i != s.size(); ++i) {
      if (s[i] == '(') {
        lefts.push(i);
      } else {
        assert(s[i] == ')');
        if (!lefts.empty()) {
          int l = lefts.top(); lefts.pop();
          l = r_to_l[l];
          length = max(length, i + 1 - l);
          r_to_l[i + 1] = l;
        }
      }
    }
    return length;
  }
};
// @lc code=end

