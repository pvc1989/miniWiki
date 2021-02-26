/*
 * @lc app=leetcode id=44 lang=cpp
 *
 * [44] Wildcard Matching
 *
 * https://leetcode.com/problems/wildcard-matching/description/
 *
 * algorithms
 * Hard (25.37%)
 * Likes:    2728
 * Dislikes: 130
 * Total Accepted:    288.6K
 * Total Submissions: 1.1M
 * Testcase Example:  '"aa"\n"a"'
 *
 * Given an input string (s) and a pattern (p), implement wildcard pattern
 * matching with support for '?' and '*' where:
 * 
 * 
 * '?' Matches any single character.
 * '*' Matches any sequence of characters (including the empty sequence).
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
 * Input: s = "aa", p = "*"
 * Output: true
 * Explanation: '*' matches any sequence.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: s = "cb", p = "?a"
 * Output: false
 * Explanation: '?' matches 'c', but the second letter is 'a', which does not
 * match 'b'.
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: s = "adceb", p = "*a*b"
 * Output: true
 * Explanation: The first '*' matches the empty sequence, while the second '*'
 * matches the substring "dce".
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: s = "acdcb", p = "a*c?b"
 * Output: false
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= s.length, p.length <= 2000
 * s contains only lowercase English letters.
 * p contains only lowercase English letters, '?' or '*'.
 * 
 * 
 */

// @lc code=start
class Solution {
  using Iter = string::iterator;
  static bool match(Iter s_head, Iter s_tail, Iter p_head, Iter p_tail) {
    if (p_head == p_tail) return s_head == s_tail;
    if (s_head == s_tail) {
      return *p_head == '*' && match(s_head, s_tail, ++p_head, p_tail);
    }
    bool is_match = false;
    switch (*p_head) {
    case '*':
      while (p_head != p_tail && *p_head == '*') {
        ++p_head;  // treat a long `*`-sequence as a single `*`
      }
      is_match = match(s_head, s_tail, p_head, p_tail);
      while (!is_match && s_head != s_tail) {
        is_match = match(++s_head, s_tail, p_head, p_tail);
      }
      break;
    case '?':
      is_match = match(++s_head, s_tail, ++p_head, p_tail);
      break;
    default:
      is_match = (*s_head == *p_head)
          && match(++s_head, s_tail, ++p_head, p_tail);
      break;
    }
    return is_match;
  }
  bool match(string const &s, string old_p) {
    replace(old_p.begin(), old_p.end(), '?', '.');
    // replace '*'-sequence by '.*'
    auto new_p = string();
    for (int i = 0; i != old_p.size(); ++i) {
      if (old_p[i] != '*') {
        new_p += old_p[i];
      } else {
        new_p += ".*";
        while (i != old_p.size() && old_p[i] == '*') {
          ++i;
        }
        --i;
      }
    }
    // cout << new_p << endl;
    return regex_match(s, regex(new_p));
  }
  static bool match(string const &s, string const &p, int i_s, int i_p) {
    auto n_s = s.size(), n_p = p.size();
    auto prev_row = vector<bool>(n_p + 1);
    auto curr_row = vector<bool>(n_p + 1);
    curr_row[0] = true;
    for (int i_p = 0; i_p < n_p && curr_row[i_p]; ++i_p) {
      curr_row[i_p + 1] = (p[i_p] == '*');
    }
    for (int i_s = 0; i_s < n_s; ++i_s) {
      swap(prev_row, curr_row);
      curr_row[0] = false;
      for (int i_p = 0; i_p < n_p; ++i_p) {
        if (p[i_p] == '*') {
          curr_row[i_p + 1] = prev_row[i_p + 1] || curr_row[i_p];
        } else {
          curr_row[i_p + 1] = prev_row[i_p]
              && (p[i_p] == s[i_s] || p[i_p] == '?');
        }
      }
    }
    return curr_row.back();
  }
 public:
  bool isMatch(string s, string p) {
    return match(s, p, 0, 0);
  }
};
// @lc code=end

