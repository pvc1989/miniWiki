/*
 * @lc app=leetcode id=28 lang=cpp
 *
 * [28] Implement strStr()
 *
 * https://leetcode.com/problems/implement-strstr/description/
 *
 * algorithms
 * Easy (35.15%)
 * Likes:    2158
 * Dislikes: 2252
 * Total Accepted:    819K
 * Total Submissions: 2.3M
 * Testcase Example:  '"hello"\n"ll"'
 *
 * Implement strStr().
 * 
 * Return the index of the first occurrence of needle in haystack, or -1 if
 * needle is not part of haystack.
 * 
 * Clarification:
 * 
 * What should we return when needle is an empty string? This is a great
 * question to ask during an interview.
 * 
 * For the purpose of this problem, we will return 0 when needle is an empty
 * string. This is consistent to C's strstr() and Java's indexOf().
 * 
 * 
 * Example 1:
 * Input: haystack = "hello", needle = "ll"
 * Output: 2
 * Example 2:
 * Input: haystack = "aaaaa", needle = "bba"
 * Output: -1
 * Example 3:
 * Input: haystack = "", needle = ""
 * Output: 0
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= haystack.length, needle.length <= 5 * 10^4
 * haystack and needle consist of only lower-case English characters.
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<vector<int>> build(string const &word) {
    auto dfa = vector<vector<int>>('z'-'a'+1, vector<int>(word.size()));
    assert(word.size());
    dfa[word[0]-'a'][0] = 1;
    int state = 0;  // the state after running the `dfa` on `word[1, i_word)`
    for (int i_word = 1; i_word != word.size(); ++i_word) {
      for (int i_char = 0; i_char != dfa.size(); ++i_char) {
        dfa[i_char][i_word] = dfa[i_char][state];  // mismatch transition
      }
      auto i_next = word[i_word] - 'a';
      dfa[i_next][i_word] = i_word + 1;  // match transition
      state = dfa[i_next][state];
    }
    return dfa;
  }
 public:
  int strStr(string text, string word) {
    if (word.size() == 0) return 0;
    auto dfa = build(word);
    int i_text = 0, i_word = 0;
    while (i_text != text.size() && i_word != word.size()) {
      // invariance: text[i_text-i_word, i_text) == word[0, i_word)
      i_word = dfa[text[i_text]-'a'][i_word];
      ++i_text;
    }
    return i_word == word.size() ? i_text-i_word : -1;
  }
};
// @lc code=end

