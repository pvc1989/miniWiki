/*
 * @lc app=leetcode id=792 lang=cpp
 *
 * [792] Number of Matching Subsequences
 *
 * https://leetcode.com/problems/number-of-matching-subsequences/description/
 *
 * algorithms
 * Medium (48.74%)
 * Likes:    1521
 * Dislikes: 92
 * Total Accepted:    64.4K
 * Total Submissions: 132.3K
 * Testcase Example:  '"abcde"\n["a","bb","acd","ace"]'
 *
 * Given a string s and an array of strings words, return the number of
 * words[i] that is a subsequence of s.
 * 
 * A subsequence of a string is a new string generated from the original string
 * with some characters (can be none) deleted without changing the relative
 * order of the remaining characters.
 * 
 * 
 * For example, "ace" is a subsequence of "abcde".
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "abcde", words = ["a","bb","acd","ace"]
 * Output: 3
 * Explanation: There are three strings in words that are a subsequence of s:
 * "a", "acd", "ace".
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s = "dsahjpjauf", words = ["ahjpjau","ja","ahbwzgqnuk","tnmlanowax"]
 * Output: 2
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 5 * 10^4
 * 1 <= words.length <= 5000
 * 1 <= words[i].length <= 50
 * s and words[i] consist of only lowercase English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int numMatchingSubseq(string s, vector<string>& words) {
    array<vector<const char*>, 128>waiting;  // ['a', 'z'] == [97, 122]
    for (auto &w : words)
      waiting[w[0]].push_back(w.c_str());
    for (char c : s) {
      vector<const char*> pending;
      swap(waiting[c], pending);
      for (const char* c_ptr : pending)
        waiting[*++c_ptr].push_back(c_ptr);
    }
    return waiting[0].size();
  }
};
// @lc code=end

