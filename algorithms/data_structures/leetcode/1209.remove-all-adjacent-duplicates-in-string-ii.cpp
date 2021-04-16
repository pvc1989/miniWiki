/*
 * @lc app=leetcode id=1209 lang=cpp
 *
 * [1209] Remove All Adjacent Duplicates in String II
 *
 * https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/description/
 *
 * algorithms
 * Medium (57.63%)
 * Likes:    1257
 * Dislikes: 30
 * Total Accepted:    76.9K
 * Total Submissions: 132.4K
 * Testcase Example:  '"abcd"\n2'
 *
 * Given a string s, a k duplicate removal consists of choosing k adjacent and
 * equal letters from s and removing them causing the left and the right side
 * of the deleted substring to concatenate together.
 * 
 * We repeatedly make k duplicate removals on s until we no longer can.
 * 
 * Return the final string after all such duplicate removals have been made.
 * 
 * It is guaranteed that the answer is unique.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "abcd", k = 2
 * Output: "abcd"
 * Explanation: There's nothing to delete.
 * 
 * Example 2:
 * 
 * 
 * Input: s = "deeedbbcccbdaa", k = 3
 * Output: "aa"
 * Explanation: 
 * First delete "eee" and "ccc", get "ddbbbdaa"
 * Then delete "bbb", get "dddaa"
 * Finally delete "ddd", get "aa"
 * 
 * Example 3:
 * 
 * 
 * Input: s = "pbbcggttciiippooaais", k = 2
 * Output: "ps"
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 10^5
 * 2 <= k <= 10^4
 * s only contains lower case English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  string removeDuplicates(string s, int k) {
    auto char_and_count = vector<pair<char, int>>();
    for (auto c : s) {
      if (char_and_count.empty() || c != char_and_count.back().first) {
        char_and_count.emplace_back(c, 1);
      } else {
        assert(c == char_and_count.back().first);
        if (++char_and_count.back().second == k)
          char_and_count.pop_back();
      }
    }
    auto result = string();
    for (auto& [c, count] : char_and_count)
      result.append(string(count, c));
    return result;
  }
};
// @lc code=end

