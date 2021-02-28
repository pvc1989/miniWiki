/*
 * @lc app=leetcode id=131 lang=cpp
 *
 * [131] Palindrome Partitioning
 *
 * https://leetcode.com/problems/palindrome-partitioning/description/
 *
 * algorithms
 * Medium (52.03%)
 * Likes:    3058
 * Dislikes: 97
 * Total Accepted:    296.9K
 * Total Submissions: 570.7K
 * Testcase Example:  '"aab"'
 *
 * Given a string s, partition s such that every substring of the partition is
 * a palindrome. Return all possible palindrome partitioning of s.
 * 
 * A palindrome string is a string that reads the same backward as forward.
 * 
 * 
 * Example 1:
 * Input: s = "aab"
 * Output: [["a","a","b"],["aa","b"]]
 * Example 2:
 * Input: s = "a"
 * Output: [["a"]]
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 16
 * s contains only lowercase English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<vector<string>> partition(string s) {
    // is_palindromic[size][head] := s[head, head + size) is a palindrome
    auto is_palindromic = array<array<bool, 17>, 17>();
    is_palindromic[0].fill(true);
    is_palindromic[1].fill(true);
    for (int size = 2; size <= s.size(); ++size) {
      for (int head = s.size() - size; 0 <= head; --head) {
        is_palindromic[size][head] = is_palindromic[size - 2][head + 1]
            && s[head] == s[head + size - 1];
      }
    }
    // dp[tail] := solution on s[0, tail)
    auto dp = vector<vector<vector<string>>>();
    for (int tail = 0; tail <= s.size(); ++tail) {  // tail of subproblem
      dp.emplace_back();
      auto& curr = dp.back();
      if (is_palindromic[tail][0]) {
        curr.emplace_back();
        curr.back().emplace_back(s.substr(0, tail));
      }
      for (int head = 1; head < tail; ++head) {  // head of last 
        int size = tail - head;
        if (is_palindromic[size][head]) {
          for (auto& prefix : dp[head]) {
            curr.emplace_back(prefix);
            curr.back().emplace_back(s.substr(head, size));
          }
        }
      }
    }
    return dp.back();
  }
};
// @lc code=end

