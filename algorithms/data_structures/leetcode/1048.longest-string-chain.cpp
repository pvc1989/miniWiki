/*
 * @lc app=leetcode id=1048 lang=cpp
 *
 * [1048] Longest String Chain
 *
 * https://leetcode.com/problems/longest-string-chain/description/
 *
 * algorithms
 * Medium (55.56%)
 * Likes:    1738
 * Dislikes: 107
 * Total Accepted:    105.9K
 * Total Submissions: 189.7K
 * Testcase Example:  '["a","b","ba","bca","bda","bdca"]'
 *
 * Given a list of words, each word consists of English lowercase letters.
 * 
 * Let's say word1 is a predecessor of word2 if and only if we can add exactly
 * one letter anywhere in word1 to make it equal to word2. For example, "abc"
 * is a predecessor of "abac".
 * 
 * A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >=
 * 1, where word_1 is a predecessor of word_2, word_2 is a predecessor of
 * word_3, and so on.
 * 
 * Return the longest possible length of a word chain with words chosen from
 * the given list of words.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: words = ["a","b","ba","bca","bda","bdca"]
 * Output: 4
 * Explanation: One of the longest word chain is "a","ba","bda","bdca".
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
 * Output: 5
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= words.length <= 1000
 * 1 <= words[i].length <= 16
 * words[i] only consists of English lowercase letters.
 * 
 * 
 */

// @lc code=start
class Solution {
  array<unordered_map<string, int>, 17> size_to_group_;
  int lookup(string const& word) {
    int n = word.size();
    auto iter = size_to_group_[n].find(word);
    if (iter == size_to_group_[n].end())
      return 0;
    int chain_length = iter->second;
    if (chain_length == 0) {
      chain_length = 1;
      if (n > 1) {
        for (int i = 0; i != n; ++i) {
          auto word_shorter = word;
          word_shorter.erase(i, 1);
          chain_length = max(chain_length, lookup(word_shorter) + 1);
        }
      }
      iter->second = chain_length;
    }
    return chain_length;
  }
 public:
  int longestStrChain(vector<string>& words) {
    for (auto& w : words)
      size_to_group_[w.size()][w] = 0;
    int max_length = 0;
    for (int size = 1; size != 17; ++size) {
      auto& group = size_to_group_[size];
      for (auto& [word, length] : group) {
        max_length = max(max_length, lookup(word));
      }
    }
    return max_length;
  }
};
// @lc code=end

