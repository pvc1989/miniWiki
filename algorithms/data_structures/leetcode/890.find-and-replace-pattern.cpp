/*
 * @lc app=leetcode id=890 lang=cpp
 *
 * [890] Find and Replace Pattern
 *
 * https://leetcode.com/problems/find-and-replace-pattern/description/
 *
 * algorithms
 * Medium (74.19%)
 * Likes:    1036
 * Dislikes: 92
 * Total Accepted:    66.4K
 * Total Submissions: 89.3K
 * Testcase Example:  '["abc","deq","mee","aqq","dkd","ccc"]\n"abb"'
 *
 * Given a list of strings words and a string pattern, return a list of
 * words[i] that match pattern. You may return the answer in any order.
 * 
 * A word matches the pattern if there exists a permutation of letters p so
 * that after replacing every letter x in the pattern with p(x), we get the
 * desired word.
 * 
 * Recall that a permutation of letters is a bijection from letters to letters:
 * every letter maps to another letter, and no two letters map to the same
 * letter.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
 * Output: ["mee","aqq"]
 * Explanation: "mee" matches the pattern because there is a permutation {a ->
 * m, b -> e, ...}. 
 * "ccc" does not match the pattern because {a -> c, b -> c, ...} is not a
 * permutation, since a and b map to the same letter.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: words = ["a","b","c"], pattern = "a"
 * Output: ["a","b","c"]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= pattern.length <= 20
 * 1 <= words.length <= 50
 * words[i].length == pattern.length
 * pattern and words[i] are lowercase English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
  bool match(string const& w, string const&p) {
    array<char, 26> w_to_p; w_to_p.fill(-1);
    array<char, 26> p_to_w; p_to_w.fill(-1);
    for (int i = p.size() - 1; i >= 0; --i) {
      int w_i = w[i] - 'a', p_i = p[i] - 'a';
      if (w_to_p[w_i] != -1) {
        if (p_to_w[p_i] == w_i)
          continue;
        else
          return false;
      }
      else {
        if (p_to_w[p_i] == -1) {
          p_to_w[p_i] = w_i;
          w_to_p[w_i] = p_i;
        }
        else {
          return false;
        }
      }
    }
    return true;
  }
 public:
  vector<string> findAndReplacePattern(vector<string>& words, string pattern) {
    vector<string> result;
    for (auto& word : words)
      if (match(word, pattern))
        result.emplace_back(word);
    return result;
  }
};
// @lc code=end

