/*
 * @lc app=leetcode id=816 lang=cpp
 *
 * [816] Ambiguous Coordinates
 *
 * https://leetcode.com/problems/ambiguous-coordinates/description/
 *
 * algorithms
 * Medium (48.12%)
 * Likes:    152
 * Dislikes: 312
 * Total Accepted:    13.2K
 * Total Submissions: 26.3K
 * Testcase Example:  '"(123)"'
 *
 * We had some 2-dimensional coordinates, like "(1, 3)" or "(2, 0.5)".  Then,
 * we removed all commas, decimal points, and spaces, and ended up with the
 * string s.  Return a list of strings representing all possibilities for what
 * our original coordinates could have been.
 * 
 * Our original representation never had extraneous zeroes, so we never started
 * with numbers like "00", "0.0", "0.00", "1.0", "001", "00.01", or any other
 * number that can be represented with less digits.  Also, a decimal point
 * within a number never occurs without at least one digit occuring before it,
 * so we never started with numbers like ".1".
 * 
 * The final answer list can be returned in any order.  Also note that all
 * coordinates in the final answer have exactly one space between them
 * (occurring after the comma.)
 * 
 * 
 * Example 1:
 * Input: s = "(123)"
 * Output: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]
 * 
 * 
 * 
 * Example 2:
 * Input: s = "(00011)"
 * Output:  ["(0.001, 1)", "(0, 0.011)"]
 * Explanation: 
 * 0.0, 00, 0001 or 00.01 are not allowed.
 * 
 * 
 * 
 * Example 3:
 * Input: s = "(0123)"
 * Output: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)",
 * "(0.12, 3)"]
 * 
 * 
 * 
 * Example 4:
 * Input: s = "(100)"
 * Output: [(10, 0)]
 * Explanation: 
 * 1.0 is not allowed.
 * 
 * 
 * 
 * 
 * Note: 
 * 
 * 
 * 4 <= s.length <= 12.
 * s[0] = "(", s[s.length - 1] = ")", and the other elements in s are
 * digits.
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<string> make(string s, int i, int j) {
    auto ans = vector<string>();
    int l = j - i;
    for (int d = 1; d <= l; ++d) {
      auto int_part = s.substr(i, d);
      auto dec_part = s.substr(i+d, l-d);
      if ((int_part == "0" || !(int_part[0] == '0')) && !(dec_part.back() == '0'))
        ans.emplace_back(int_part + (d < l ? "." : "") + dec_part);
    }
    return ans;
  }
 public:
  vector<string> ambiguousCoordinates(string s) {
    auto n = s.size();
    auto ans = vector<string>();
    for (int i = 2; i < n - 1; ++i)
        for (auto x : make(s, 1, i))
            for (auto y : make(s, i, n - 1))
                ans.emplace_back("(" + x + ", " + y + ")");
    return ans;
  }
};
// @lc code=end

