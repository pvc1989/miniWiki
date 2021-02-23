/*
 * @lc app=leetcode id=76 lang=cpp
 *
 * [76] Minimum Window Substring
 *
 * https://leetcode.com/problems/minimum-window-substring/description/
 *
 * algorithms
 * Hard (35.90%)
 * Likes:    6062
 * Dislikes: 409
 * Total Accepted:    499.2K
 * Total Submissions: 1.4M
 * Testcase Example:  '"ADOBECODEBANC"\n"ABC"'
 *
 * Given two strings s and t, return the minimum window in s which will contain
 * all the characters in t. If there is no such window in s that covers all
 * characters in t, return the empty string "".
 * 
 * Note that If there is such a window, it is guaranteed that there will always
 * be only one unique minimum window in s.
 * 
 * 
 * Example 1:
 * Input: s = "ADOBECODEBANC", t = "ABC"
 * Output: "BANC"
 * Example 2:
 * Input: s = "a", t = "a"
 * Output: "a"
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length, t.length <= 10^5
 * s and t consist of English letters.
 * 
 * 
 * 
 * Follow up: Could you find an algorithm that runs in O(n) time?
 */

// @lc code=start
class Solution {
  array<int, 58> char_to_count_in_w_;
  array<int, 58> char_to_count_in_t_;
  vector<int> used_indices_;
  bool cover() const {
    for (int i : used_indices_) {
      if (char_to_count_in_t_[i] > char_to_count_in_w_[i]) {
        return false;
      }
    }
    return true;
  }
 public:
  string minWindow(string s, string t) {
    int head{0}, min_size{INT_MAX};  // to build solution
    // initialization
    for (char c : t) {
      ++char_to_count_in_t_[c - 'A'];
    }
    for (int i = 0; i != 58; ++i) {
      if (char_to_count_in_t_[i]) {
        used_indices_.push_back(i);
      }
    }
    // run through s
    int fast{0}, slow{0};
    while (fast != s.size()) {
      ++char_to_count_in_w_[s[fast++] - 'A'];
      while (slow < fast && cover()) {  // s[slow, fast) can cover t
        auto size = fast - slow;
        if (size < min_size) {
          head = slow;
          min_size = size;
        }
        --char_to_count_in_w_[s[slow++] - 'A'];
      }
    }
    return min_size == INT_MAX ? "" : s.substr(head, min_size);
  }
};
// @lc code=end

